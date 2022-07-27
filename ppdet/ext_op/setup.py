import os
import glob
import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup


def get_extensions():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ext_root_dir = os.path.join(root_dir, 'csrc')
    sources = []
    for ext_name in os.listdir(ext_root_dir):
        ext_dir = os.path.join(ext_root_dir, ext_name)
        source = glob.glob(os.path.join(ext_dir, '*.cc'))
        kwargs = dict()
        if paddle.device.is_compiled_with_cuda():
            source += glob.glob(os.path.join(ext_dir, '*.cu'))

        if not source:
            continue

        sources += source

    if paddle.device.is_compiled_with_cuda():
        extension = CUDAExtension(
            sources, extra_compile_args={'cxx': ['-DPADDLE_WITH_CUDA']})
    else:
        extension = CppExtension(sources)

    return extension


if __name__ == "__main__":
    setup(name='ext_op', ext_modules=get_extensions())
