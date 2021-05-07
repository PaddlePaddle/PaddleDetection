from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

if __name__ == "__main__":
    setup(
        name='rbox_iou_ops',
        ext_modules=CUDAExtension(sources=['rbox_iou_op.cc', 'rbox_iou_op.cu']))
