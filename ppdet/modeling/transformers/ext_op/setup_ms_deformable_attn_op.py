from paddle.utils.cpp_extension import CUDAExtension, setup

if __name__ == "__main__":
    setup(
        name='deformable_detr_ops',
        ext_modules=CUDAExtension(
            sources=['ms_deformable_attn_op.cc', 'ms_deformable_attn_op.cu']))
