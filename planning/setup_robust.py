import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup
import pybind11
import subprocess
import os

def get_cuda_version():
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split('.')[:2])
        return major, minor
    return 11, 8

def get_compute_capabilities():
    if torch.cuda.is_available():
        caps = []
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            caps.append(f"{major}.{minor}")
        return caps or ["8.9"]
    return ["8.9"]

def build_nvcc_gencode_flags(capabilities):
    gencode_flags = []
    for cap in capabilities:
        cap_num = cap.replace('.', '')
        gencode_flags.append(f'-gencode=arch=compute_{cap_num},code=sm_{cap_num}')
    highest_cap = max(capabilities, key=lambda x: float(x))
    highest_cap_num = highest_cap.replace('.', '')
    gencode_flags.append(f'-gencode=arch=compute_{highest_cap_num},code=compute_{highest_cap_num}')
    return gencode_flags

cuda_major, cuda_minor = get_cuda_version()
compute_caps = get_compute_capabilities()
gencode_flags = build_nvcc_gencode_flags(compute_caps)

print(f"Building for CUDA {cuda_major}.{cuda_minor}")
print(f"Target compute capabilities: {compute_caps}")

nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-std=c++17',
    '-rdc=true',  # allow cross-file device calls
    '-DTORCH_EXTENSION_NAME=cuPRM',
    '-DWITH_CUDA',
    '--compiler-options=-fPIC',
]
nvcc_flags.extend(gencode_flags)

cxx_flags = [
    '-O3',
    '-std=c++17',
    '-DTORCH_EXTENSION_NAME=cuPRM',
    '-DWITH_CUDA',
    '-fPIC',
]

# All .cu sources that need device linking
cuda_sources = [
    'src/collision/cc_2D.cu',
    'src/collision/env_2D.cu',
    'src/planning/construction.cu',
    'src/planning/pprm.cu',
    'src/local_planning/reedsshepp.cu',
]

# Create build directory if missing
os.makedirs("build", exist_ok=True)
dlink_obj = os.path.join("build", "cuda_dlink.o")

# Run nvcc -dlink manually
nvcc_cmd = [
    "nvcc", "-dlink", "-o", dlink_obj, "-rdc=true",
    "-Xcompiler", "-fPIC"
] + cuda_sources + gencode_flags + ["-lcudadevrt", "-lcudart_static"]

print("Running device link step:", " ".join(nvcc_cmd))
subprocess.run(nvcc_cmd, check=True)

ext_modules = [
    CUDAExtension(
        name="cuPRM",
        sources=["src/bindings/py_bind.cpp"] + cuda_sources,
        include_dirs=["src/", pybind11.get_include()],
        extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags},
        libraries=["cudart", "curand"],
        extra_link_args=["-lcudadevrt", "-lcudart_static"],
        extra_objects=[dlink_obj],  # include device link object
    )
]

setup(
    name="cuPRM",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
)
