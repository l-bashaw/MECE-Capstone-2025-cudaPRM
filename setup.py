from setuptools import setup
import torch
from torch.utils import cpp_extension
import pybind11

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install CUDA-enabled PyTorch.")

# Define the extension
ext_modules = [
    cpp_extension.CUDAExtension(
        name='parallelrm_cuda',
        sources=[
            'src/python_bindings.cpp',
            'src/collision/cc_2D.cu', 
            'src/collision/env_2D.cu',
            'src/planning/construction.cu',
            'src/planning/pprm.cu',
        ],
        include_dirs=[
            'src/',
            pybind11.get_include(),
        ],
        extra_compile_args={
            'cxx': [
                '-O3', 
                '-std=c++17',
                '-DTORCH_EXTENSION_NAME=parallelrm_cuda',
                '-DWITH_CUDA',
                '-fPIC'
            ],
            'nvcc': [
                '-O3', 
                '--use_fast_math', 
                '-gencode=arch=compute_86,code=sm_86',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '-std=c++17',
                #'-rdc=true',
                '-DTORCH_EXTENSION_NAME=parallelrm_cuda',
                '-DWITH_CUDA',
                '--compiler-options=-fPIC'
            ]
        },
        libraries=['cudart', 'curand'],  # Remove 'cudadevrt'
        language='c++',
    )
]

setup(
    name='parallelrm',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
)