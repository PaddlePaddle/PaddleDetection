//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "keypoint_postprocess.h"
#define PI 3.1415926535
#define HALF_CIRCLE_DEGREE 180

cv::Point2f get_3rd_point(cv::Point2f& a, cv::Point2f& b) {
  cv::Point2f direct{a.x - b.x, a.y - b.y};
  return cv::Point2f(a.x - direct.y, a.y + direct.x);
}

std::vector<float> get_dir(float src_point_x,
                           float src_point_y,
                           float rot_rad) {
  float sn = sin(rot_rad);
  float cs = cos(rot_rad);
  std::vector<float> src_result{0.0, 0.0};
  src_result[0] = src_point_x * cs - src_point_y * sn;
  src_result[1] = src_point_x * sn + src_point_y * cs;
  return src_result;
}

void affine_tranform(
    float pt_x, float pt_y, cv::Mat& trans, std::vector<float>& preds, int p) {
  double new1[3] = {pt_x, pt_y, 1.0};
  cv::Mat new_pt(3, 1, trans.type(), new1);
  cv::Mat w = trans * new_pt;
  preds[p * 3 + 1] = static_cast<float>(w.at<double>(0, 0));
  preds[p * 3 + 2] = static_cast<float>(w.at<double>(1, 0));
}

void get_affine_transform(std::vector<float>& center,
                          std::vector<float>& scale,
                          float rot,
                          std::vector<int>& output_size,
                          cv::Mat& trans,
                          int inv) {
  float src_w = scale[0];
  float dst_w = static_cast<float>(output_size[0]);
  float dst_h = static_cast<float>(output_size[1]);
  float rot_rad = rot * PI / HALF_CIRCLE_DEGREE;
  std::vector<float> src_dir = get_dir(-0.5 * src_w, 0, rot_rad);
  std::vector<float> dst_dir{static_cast<float>(-0.5) * dst_w, 0.0};
  cv::Point2f srcPoint2f[3], dstPoint2f[3];
  srcPoint2f[0] = cv::Point2f(center[0], center[1]);
  srcPoint2f[1] = cv::Point2f(center[0] + src_dir[0], center[1] + src_dir[1]);
  srcPoint2f[2] = get_3rd_point(srcPoint2f[0], srcPoint2f[1]);

  dstPoint2f[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
  dstPoint2f[1] =
      cv::Point2f(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);
  dstPoint2f[2] = get_3rd_point(dstPoint2f[0], dstPoint2f[1]);
  if (inv == 0) {
    trans = cv::getAffineTransform(srcPoint2f, dstPoint2f);
  } else {
    trans = cv::getAffineTransform(dstPoint2f, srcPoint2f);
  }
}

void transform_preds(std::vector<float>& coords,
                     std::vector<float>& center,
                     std::vector<float>& scale,
                     std::vector<int>& output_size,
                     std::vector<uint64_t>& dim,
                     std::vector<float>& target_coords,
                     bool affine=false) {
  if (affine) {
    cv::Mat trans(2, 3, CV_64FC1);
    get_affine_transform(center, scale, 0, output_size, trans, 1);
    for (int p = 0; p < dim[1]; ++p) {
      affine_tranform(
          coords[p * 2], coords[p * 2 + 1], trans, target_coords, p);
    }
  } else {
    float heat_w = static_cast<float>(output_size[0]);
    float heat_h = static_cast<float>(output_size[1]);
    float x_scale = scale[0] / heat_w;
    float y_scale = scale[1] / heat_h;
    float offset_x = center[0] - scale[0] / 2.;
    float offset_y = center[1] - scale[1] / 2.;
    for (int i = 0; i < dim[1]; i++) {
      target_coords[i * 3 + 1] = x_scale * coords[i * 2] + offset_x;
      target_coords[i * 3 + 2] = y_scale * coords[i * 2 + 1] + offset_y;
    }
  }
}

// only for batchsize == 1
void get_max_preds(std::vector<float>& heatmap,
                   std::vector<int>& dim,
                   std::vector<float>& preds,
                   std::vector<float>& maxvals,
                   int batchid,
                   int joint_idx) {
  int num_joints = dim[1];
  int width = dim[3];
  std::vector<int> idx;
  idx.resize(num_joints * 2);

  for (int j = 0; j < dim[1]; j++) {
    float* index = &(
        heatmap[batchid * num_joints * dim[2] * dim[3] + j * dim[2] * dim[3]]);
    float* end = index + dim[2] * dim[3];
    float* max_dis = std::max_element(index, end);
    auto max_id = std::distance(index, max_dis);
    maxvals[j] = *max_dis;
    if (*max_dis > 0) {
      preds[j * 2] = static_cast<float>(max_id % width);
      preds[j * 2 + 1] = static_cast<float>(max_id / width);
    }
  }
}

void dark_parse(std::vector<float>& heatmap,
                std::vector<uint64_t>& dim,
                std::vector<float>& coords,
                int px,
                int py,
                int index,
                int ch) {
  /*DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
  Representation for Human Pose Estimation (CVPR 2020).
  1) offset = - hassian.inv() * derivative
  2) dx = (heatmap[x+1] - heatmap[x-1])/2.
  3) dxx = (dx[x+1] - dx[x-1])/2.
  4) derivative = Mat([dx, dy])
  5) hassian = Mat([[dxx, dxy], [dxy, dyy]])
  */
  std::vector<float>::const_iterator first1 = heatmap.begin() + index;
  std::vector<float>::const_iterator last1 =
      heatmap.begin() + index + dim[2] * dim[3];
  std::vector<float> heatmap_ch(first1, last1);
  cv::Mat heatmap_mat = cv::Mat(heatmap_ch).reshape(0, dim[2]);
  heatmap_mat.convertTo(heatmap_mat, CV_32FC1);
  cv::GaussianBlur(heatmap_mat, heatmap_mat, cv::Size(3, 3), 0, 0);
  heatmap_mat = heatmap_mat.reshape(1, 1);
  heatmap_ch = std::vector<float>(heatmap_mat.reshape(1, 1));

  float epsilon = 1e-10;
  // sample heatmap to get values in around target location
  float xy = log(fmax(heatmap_ch[py * dim[3] + px], epsilon));
  float xr = log(fmax(heatmap_ch[py * dim[3] + px + 1], epsilon));
  float xl = log(fmax(heatmap_ch[py * dim[3] + px - 1], epsilon));

  float xr2 = log(fmax(heatmap_ch[py * dim[3] + px + 2], epsilon));
  float xl2 = log(fmax(heatmap_ch[py * dim[3] + px - 2], epsilon));
  float yu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px], epsilon));
  float yd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px], epsilon));
  float yu2 = log(fmax(heatmap_ch[(py + 2) * dim[3] + px], epsilon));
  float yd2 = log(fmax(heatmap_ch[(py - 2) * dim[3] + px], epsilon));
  float xryu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px + 1], epsilon));
  float xryd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px + 1], epsilon));
  float xlyu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px - 1], epsilon));
  float xlyd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px - 1], epsilon));

  // compute dx/dy and dxx/dyy with sampled values
  float dx = 0.5 * (xr - xl);
  float dy = 0.5 * (yu - yd);
  float dxx = 0.25 * (xr2 - 2 * xy + xl2);
  float dxy = 0.25 * (xryu - xryd - xlyu + xlyd);
  float dyy = 0.25 * (yu2 - 2 * xy + yd2);

  // finally get offset by derivative and hassian, which combined by dx/dy and
  // dxx/dyy
  if (dxx * dyy - dxy * dxy != 0) {
    float M[2][2] = {dxx, dxy, dxy, dyy};
    float D[2] = {dx, dy};
    cv::Mat hassian(2, 2, CV_32F, M);
    cv::Mat derivative(2, 1, CV_32F, D);
    cv::Mat offset = -hassian.inv() * derivative;
    coords[ch * 2] += offset.at<float>(0, 0);
    coords[ch * 2 + 1] += offset.at<float>(1, 0);
  }
}

void get_final_preds(std::vector<float>& heatmap,
                     std::vector<uint64_t>& dim,
                     std::vector<float>& idxout,
                     std::vector<uint64_t>& idxdim,
                     std::vector<float>& center,
                     std::vector<float> scale,
                     std::vector<float>& preds,
                     int batchid,
                     bool DARK) {
  std::vector<float> coords;
  coords.resize(dim[1] * 2);
  int heatmap_height = dim[2];
  int heatmap_width = dim[3];

  for (int j = 0; j < dim[1]; ++j) {
    int index = (batchid * dim[1] + j) * dim[2] * dim[3];

    int idx = int(idxout[batchid * dim[1] + j]);
    preds[j * 3] = heatmap[index + idx];
    coords[j * 2] = idx % heatmap_width;
    coords[j * 2 + 1] = idx / heatmap_width;

    int px = int(coords[j * 2] + 0.5);
    int py = int(coords[j * 2 + 1] + 0.5);

    if (DARK && px > 1 && px < heatmap_width - 2 && py > 1 &&
        py < heatmap_height - 2) {
      dark_parse(heatmap, dim, coords, px, py, index, j);
    } else {
      if (px > 0 && px < heatmap_width - 1) {
        float diff_x = heatmap[index + py * dim[3] + px + 1] -
                       heatmap[index + py * dim[3] + px - 1];
        coords[j * 2] += diff_x > 0 ? 1 : -1 * 0.25;
      }
      if (py > 0 && py < heatmap_height - 1) {
        float diff_y = heatmap[index + (py + 1) * dim[3] + px] -
                       heatmap[index + (py - 1) * dim[3] + px];
        coords[j * 2 + 1] += diff_y > 0 ? 1 : -1 * 0.25;
      }
    }
  }

  std::vector<int> img_size{heatmap_width, heatmap_height};
  transform_preds(coords, center, scale, img_size, dim, preds);
}

void CropImg(cv::Mat& img,
             cv::Mat& crop_img,
             std::vector<int>& area,
             std::vector<float>& center,
             std::vector<float>& scale,
             float expandratio) {
  int crop_x1 = std::max(0, area[0]);
  int crop_y1 = std::max(0, area[1]);
  int crop_x2 = std::min(img.cols - 1, area[2]);
  int crop_y2 = std::min(img.rows - 1, area[3]);

  int center_x = (crop_x1 + crop_x2) / 2.;
  int center_y = (crop_y1 + crop_y2) / 2.;
  int half_h = (crop_y2 - crop_y1) / 2.;
  int half_w = (crop_x2 - crop_x1) / 2.;

  if (half_h * 3 > half_w * 4) {
    half_w = static_cast<int>(half_h * 0.75);
  } else {
    half_h = static_cast<int>(half_w * 4 / 3);
  }

  crop_x1 =
      std::max(0, center_x - static_cast<int>(half_w * (1 + expandratio)));
  crop_y1 =
      std::max(0, center_y - static_cast<int>(half_h * (1 + expandratio)));
  crop_x2 = std::min(img.cols - 1,
                     static_cast<int>(center_x + half_w * (1 + expandratio)));
  crop_y2 = std::min(img.rows - 1,
                     static_cast<int>(center_y + half_h * (1 + expandratio)));
  crop_img =
      img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));

  center.clear();
  center.emplace_back((crop_x1 + crop_x2) / 2);
  center.emplace_back((crop_y1 + crop_y2) / 2);
  scale.clear();
  scale.emplace_back((crop_x2 - crop_x1));
  scale.emplace_back((crop_y2 - crop_y1));
}
