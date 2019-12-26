#include <cmath>
#include <algorithm>

namespace nms {
struct proposal {
  float score, x1, y1, x2, y2;
};

inline static bool cmp(const proposal& a, const proposal& b) {
  return a.score < b.score;
}

inline static float iou(const proposal&, const proposal&)
    __attribute__((always_inline));

static float iou(const proposal& a, const proposal& b) {
  auto overlap = 0.f;
  float iw = std::min(b.x2, a.x2) - std::max(b.x1, a.x1) + 1;
  if (iw > 0) {
    float ih = std::min(b.y2, a.y2) - std::max(b.y1, a.y1) + 1;
    if (ih > 0) {
      float ab = (b.x2 - b.x1 + 1) * (b.y2 - b.y1 + 1);
      float aa = (a.x2 - a.x1 + 1) * (a.y2 - a.y1 + 1);
      float inter = iw * ih;
      overlap = inter / (aa + ab - inter);
    }
  }
  return overlap;
}

enum class Method : uint32_t { LINEAR = 0, GAUSSIAN, HARD };

size_t soft_nms(float* boxes, int32_t* index, size_t count, Method method,
                float Nt, float sigma, float threshold) {
  std::iota(index, index + count, 0);  // np.arange()
  auto p = reinterpret_cast<proposal*>(boxes);

  auto N = count;
  for (size_t i = 0; i < N; ++i) {
    auto max = std::max_element(p + i, p + N, cmp);
    std::swap(p[i], *max);
    std::swap(index[i], index[max - p]);

    auto j = i + 1;
    auto weight = 0.f;
    while (j < N) {
      auto ov = iou(p[i], p[j]);
      switch (method) {
        case Method::LINEAR:
          weight = ov > Nt ? 1.f - ov : 1.f;
          break;
        case Method::GAUSSIAN:
          weight = std::exp(-(ov * ov) / sigma);
          break;
        case Method::HARD:
          weight = ov > Nt ? 0.f : 1.f;
          break;
      }
      p[j].score *= weight;
      if (p[j].score < threshold) {
        N--;
        std::swap(p[j], p[N]);
        std::swap(index[j], index[N]);
        j--;
      }
      j++;
    }
  };

  return N;
}
} /* namespace nms */
