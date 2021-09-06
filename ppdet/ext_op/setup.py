import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

if __name__ == "__main__":
    if paddle.device.is_compiled_with_cuda():
        setup(
            name='rbox_iou_ops',
            ext_modules=CUDAExtension(
                sources=['rbox_iou_op.cc', 'rbox_iou_op.cu'],
                extra_compile_args={'cxx': ['-DPADDLE_WITH_CUDA']}))
    else:
        setup(
            name='rbox_iou_ops',
            ext_modules=CppExtension(sources=['rbox_iou_op.cc']))
