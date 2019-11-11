#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace nms
{
enum class Method : uint32_t
{
  LINEAR = 0,
  GAUSSIAN,
  HARD
};

size_t soft_nms(float* boxes,
                int32_t* index,
                size_t count,
                Method method,
                float Nt,
                float sigma,
                float threshold);
}  // namespace nms

namespace binding
{
namespace py = pybind11;
using namespace pybind11::literals;

py::tuple py_soft_nms(py::array_t<float, py::array::c_style> boxes,
                      nms::Method method = nms::Method::GAUSSIAN,
                      float Nt           = 0.3,
                      float sigma        = 0.5,
                      float threshold    = 0.001)
{
  assert(boxes.ndim() == 2 && "Input should be 2-D NumPy array");
  assert(boxes.shape()[1] == 5 && "Input should have size [N,5]");

  auto count = boxes.size() / 5;
  auto i = new int32_t[count];
  auto b = new float[boxes.size()];
  std::copy(boxes.data(), boxes.data() + boxes.size(), b);

  auto N = nms::soft_nms(b, i, count, method, Nt, sigma, threshold);

  std::vector<size_t> shape5    = {N, 5};
  std::vector<size_t> shape1    = {N};
  std::vector<ssize_t> strides5 = {sizeof(float) * 5, sizeof(float)};
  std::vector<ssize_t> strides1 = {sizeof(float)};

  auto cap_b =
      py::capsule(b, [](void* v) { delete[] reinterpret_cast<float*>(v); });
  auto cap_i =
      py::capsule(i, [](void* v) { delete[] reinterpret_cast<int32_t*>(v); });

  auto pyb = py::array(py::dtype("float32"), shape5, strides5, b, cap_b);
  auto pyi = py::array(py::dtype("int32"), shape1, strides1, i, cap_i);
  return py::make_tuple(pyb, pyi);
}

PYBIND11_MODULE(nms, m) {
  m.doc() = "SoftNMS for object detection.";

  py::enum_<nms::Method>(m, "NMSMethod")
    .value("LINEAR", nms::Method::LINEAR)
    .value("GAUSSIAN", nms::Method::GAUSSIAN)
    .value("HARD", nms::Method::HARD)
    .export_values();
  m.def("soft_nms", &py_soft_nms, "boxes"_a.noconvert(),
        "method"_a = nms::Method::GAUSSIAN,
        "Nt"_a = 0.3, "sigma"_a = 0.5, "threshold"_a = 0.001);
}
} /* namespace binding */
