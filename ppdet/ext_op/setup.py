from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup
import paddle

if __name__ == "__main__":
    if paddle.is_compiled_with_cuda():
        setup(
            name='rbox_iou_ops',
            ext_modules=CUDAExtension(
                sources=['rbox_iou_op.cc', 'rbox_iou_op.cu'],
                include_dirs=['./']))
    else:
        setup(
            name='rbox_iou_ops',
            ext_modules=CppExtension(
                sources=['rbox_iou_op.cc'], include_dirs=['./']))
