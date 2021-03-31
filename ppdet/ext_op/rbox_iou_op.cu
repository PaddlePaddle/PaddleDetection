/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include <cassert>
#include <cmath>

#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
#else
#include <algorithm>
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
#endif

#include "paddle/extension.h"

#include <vector>

namespace {

template <typename T>
struct RotatedBox {
  T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct Point {
  T x, y;
  HOST_DEVICE_INLINE Point(const T& px = 0, const T& py = 0) : x(px), y(py) {}
  HOST_DEVICE_INLINE Point operator+(const Point& p) const {
    return Point(x + p.x, y + p.y);
  }
  HOST_DEVICE_INLINE Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  HOST_DEVICE_INLINE Point operator-(const Point& p) const {
    return Point(x - p.x, y - p.y);
  }
  HOST_DEVICE_INLINE Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
HOST_DEVICE_INLINE T dot_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T>
HOST_DEVICE_INLINE T cross_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.y - B.x * A.y;
}

template <typename T>
HOST_DEVICE_INLINE void get_rotated_vertices(
    const RotatedBox<T>& box,
    Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  //double theta = box.a * 0.01745329251;
  //MODIFIED
  double theta = box.a;
  T cosTheta2 = (T)cos(theta) * 0.5f;
  T sinTheta2 = (T)sin(theta) * 0.5f;

  // y: top --> down; x: left --> right
  pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;
}

template <typename T>
HOST_DEVICE_INLINE int get_intersection_points(
    const Point<T> (&pts1)[4],
    const Point<T> (&pts2)[4],
    Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0; // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      T det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / det;
      T t2 = cross_2d<T>(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
HOST_DEVICE_INLINE int convex_hull_graham(
    const Point<T> (&p)[24],
    const int& num_in,
    Point<T> (&q)[24],
    bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

#ifdef __CUDACC__
  // CUDA version
  // In the future, we can potentially use thrust
  // for sorting here to improve speed (though not guaranteed)
  for (int i = 1; i < num_in - 1; i++) {
    for (int j = i + 1; j < num_in; j++) {
      T crossProduct = cross_2d<T>(q[i], q[j]);
      if ((crossProduct < -1e-6) ||
          (fabs(crossProduct) < 1e-6 && dist[i] > dist[j])) {
        auto q_tmp = q[i];
        q[i] = q[j];
        q[j] = q_tmp;
        auto dist_tmp = dist[i];
        dist[i] = dist[j];
        dist[j] = dist_tmp;
      }
    }
  }
#else
  // CPU version
  std::sort(
      q + 1, q + num_in, [](const Point<T>& A, const Point<T>& B) -> bool {
        T temp = cross_2d<T>(A, B);
        if (fabs(temp) < 1e-6) {
          return dot_2d<T>(A, A) < dot_2d<T>(B, B);
        } else {
          return temp > 0;
        }
      });
#endif

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 points in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross_2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
HOST_DEVICE_INLINE T polygon_area(const Point<T> (&q)[24], const int& m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
HOST_DEVICE_INLINE T rboxes_intersection(
    const RotatedBox<T>& box1,
    const RotatedBox<T>& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}

} // namespace

template <typename T>
HOST_DEVICE_INLINE T
rbox_iou_single(T const* const box1_raw, T const* const box2_raw) {
  // shift center to the middle point to achieve higher precision in result
  RotatedBox<T> box1, box2;
  auto center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0;
  auto center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0;
  box1.x_ctr = box1_raw[0] - center_shift_x;
  box1.y_ctr = box1_raw[1] - center_shift_y;
  box1.w = box1_raw[2];
  box1.h = box1_raw[3];
  box1.a = box1_raw[4];
  box2.x_ctr = box2_raw[0] - center_shift_x;
  box2.y_ctr = box2_raw[1] - center_shift_y;
  box2.w = box2_raw[2];
  box2.h = box2_raw[3];
  box2.a = box2_raw[4];

  const T area1 = box1.w * box1.h;
  const T area2 = box2.w * box2.h;
  if (area1 < 1e-14 || area2 < 1e-14) {
    return 0.f;
  }

  const T intersection = rboxes_intersection<T>(box1, box2);
  const T iou = intersection / (area1 + area2 - intersection);
  return iou;
}


// 2D block with 32 * 16 = 512 threads per block
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T CeilDiv0(T a, T b) {
  return (a + b - 1) / b;
}

static inline int CeilDiv(const int a, const int b) {
  return (a + b -1)  / b;
}

template <typename T>
__global__ void rbox_iou_cuda_kernel(
    const int rbox1_num,
    const int rbox2_num,
    const T* rbox1_data_ptr,
    const T* rbox2_data_ptr,
    T* output_data_ptr) {

  // get row_start and col_start
  const int rbox1_block_idx = blockIdx.x * blockDim.x;
  const int rbox2_block_idx = blockIdx.y * blockDim.y;

  const int rbox1_thread_num = min(rbox1_num - rbox1_block_idx, blockDim.x);
  const int rbox2_thread_num = min(rbox2_num - rbox2_block_idx, blockDim.y);

  __shared__ T block_boxes1[BLOCK_DIM_X * 5];
  __shared__ T block_boxes2[BLOCK_DIM_Y * 5];


  // It's safe to copy using threadIdx.x since BLOCK_DIM_X >= BLOCK_DIM_Y
  if (threadIdx.x < rbox1_thread_num && threadIdx.y == 0) {
    block_boxes1[threadIdx.x * 5 + 0] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 0];
    block_boxes1[threadIdx.x * 5 + 1] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 1];
    block_boxes1[threadIdx.x * 5 + 2] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 2];
    block_boxes1[threadIdx.x * 5 + 3] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 3];
    block_boxes1[threadIdx.x * 5 + 4] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 4];
  }

  // threadIdx.x < BLOCK_DIM_Y=rbox2_thread_num, just use same condition as above: threadIdx.y == 0
  if (threadIdx.x < rbox2_thread_num && threadIdx.y == 0) {
    block_boxes2[threadIdx.x * 5 + 0] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 0];
    block_boxes2[threadIdx.x * 5 + 1] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 1];
    block_boxes2[threadIdx.x * 5 + 2] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 2];
    block_boxes2[threadIdx.x * 5 + 3] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 3];
    block_boxes2[threadIdx.x * 5 + 4] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 4];
  }

  // sync
  __syncthreads();

  if (threadIdx.x < rbox1_thread_num && threadIdx.y < rbox2_thread_num) {
    int offset = (rbox1_block_idx + threadIdx.x) * rbox2_num + rbox2_block_idx + threadIdx.y;
    output_data_ptr[offset] = rbox_iou_single<T>(block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.y * 5);
  }
}

#define CHECK_INPUT_GPU(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> RboxIouCUDAForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2) {
    CHECK_INPUT_GPU(rbox1);
    CHECK_INPUT_GPU(rbox2);

    auto rbox1_num = rbox1.shape()[0];
    auto rbox2_num = rbox2.shape()[0];

    auto output = paddle::Tensor(paddle::PlaceType::kGPU);
    output.reshape({rbox1_num, rbox2_num});

    const int blocks_x = CeilDiv(rbox1_num, BLOCK_DIM_X);
    const int blocks_y = CeilDiv(rbox2_num, BLOCK_DIM_Y);

    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    PD_DISPATCH_FLOATING_TYPES(
        rbox1.type(),
        "rbox_iou_cuda_kernel",
        ([&] {
            rbox_iou_cuda_kernel<data_t><<<blocks, threads, 0, rbox1.stream()>>>(
                rbox1_num,
                rbox2_num,
                rbox1.data<data_t>(),
                rbox2.data<data_t>(),
                output.mutable_data<data_t>());
        }));

    return {output};
}


template <typename T>
void rbox_iou_cpu_kernel(
    const int rbox1_num,
    const int rbox2_num,
    const T* rbox1_data_ptr,
    const T* rbox2_data_ptr,
    T* output_data_ptr) {

    int i, j;
    for (i = 0; i < rbox1_num; i++) {
        for (j = 0; j < rbox2_num; j++) {
		int offset = i * rbox2_num + j;
		output_data_ptr[offset] = rbox_iou_single<T>(rbox1_data_ptr + i * 5, rbox2_data_ptr + j * 5);
        }
    }
}


#define CHECK_INPUT_CPU(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> RboxIouCPUForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2) {
    CHECK_INPUT_CPU(rbox1);
    CHECK_INPUT_CPU(rbox2);

    auto rbox1_num = rbox1.shape()[0];
    auto rbox2_num = rbox2.shape()[0];

    auto output = paddle::Tensor(paddle::PlaceType::kCPU);
    output.reshape({rbox1_num, rbox2_num});

    PD_DISPATCH_FLOATING_TYPES(
        rbox1.type(),
        "rbox_iou_cpu_kernel",
        ([&] {
            rbox_iou_cpu_kernel<data_t>(
                rbox1_num,
                rbox2_num,
                rbox1.data<data_t>(),
                rbox2.data<data_t>(),
                output.mutable_data<data_t>());
        }));
    
    return {output};
}
