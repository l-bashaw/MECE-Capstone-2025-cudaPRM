import torch
from torch.utils import cpp_extension
from setuptools import setup
import pybind11
import subprocess

def get_cuda_version():
    """Get CUDA version from nvcc or torch"""
    try:
        # Try to get CUDA version from nvcc
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        for line in output.split('\n'):
            if 'release' in line:
                version_str = line.split('release')[1].split(',')[0].strip()
                major, minor = map(int, version_str.split('.'))
                return major, minor
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        if cuda_version:
            major, minor = map(int, cuda_version.split('.')[:2])
            return major, minor
    
    # Default 
    return 11, 8

def get_compute_capabilities():
    """Determine compute capabilities based on available GPUs and CUDA version"""
    cuda_major, cuda_minor = get_cuda_version()
    cuda_version = cuda_major * 10 + cuda_minor
    
    # RTX 30xx series: compute capability 8.6
    # RTX 40xx series: compute capability 8.9
    base_capabilities = ['8.6', '8.9']
    
    # Add older capabilities for broader compatibility
    if cuda_version >= 118:  # CUDA 11.8+
        capabilities = ['7.5', '8.0', '8.6', '8.9']
        
    # Add newer capabilities if CUDA version supports them
    if cuda_version >= 120:  # CUDA 12.0+
        capabilities.append('9.0')
    
    # Try to detect actual GPU capabilities 
    try:
        if torch.cuda.is_available():
            detected_caps = []
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                cap = f"{major}.{minor}"
                if cap not in detected_caps:
                    detected_caps.append(cap)
            
            # Use detected capabilities if they're reasonable
            if detected_caps and all(float(cap) >= 7.5 for cap in detected_caps):
                print(f"Detected GPU compute capabilities: {detected_caps}")
                return detected_caps
    except Exception as e:
        print(f"Could not detect GPU capabilities: {e}")
    
    print(f"Using default compute capabilities for CUDA {cuda_major}.{cuda_minor}: {capabilities}")
    return capabilities

def build_nvcc_gencode_flags(capabilities):
    """Build gencode flags for nvcc"""
    gencode_flags = []
    
    for cap in capabilities:
        # For each capability, generate both sm_XX and compute_XX
        cap_num = cap.replace('.', '')
        gencode_flags.append(f'-gencode=arch=compute_{cap_num},code=sm_{cap_num}')
    
    # Add the highest capability as compute_ for forward compatibility
    if capabilities:
        highest_cap = max(capabilities, key=lambda x: float(x))
        highest_cap_num = highest_cap.replace('.', '')
        gencode_flags.append(f'-gencode=arch=compute_{highest_cap_num},code=compute_{highest_cap_num}')
    
    return gencode_flags

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install CUDA-enabled PyTorch.")

# Get CUDA version and compute capabilities
cuda_major, cuda_minor = get_cuda_version()
compute_caps = get_compute_capabilities()
gencode_flags = build_nvcc_gencode_flags(compute_caps)

print(f"Building for CUDA {cuda_major}.{cuda_minor}")
print(f"Target compute capabilities: {compute_caps}")

# Base NVCC flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-std=c++17',
    '-DTORCH_EXTENSION_NAME=cuPRM',
    '-DWITH_CUDA',
    '--compiler-options=-fPIC'
]

# Add gencode flags
nvcc_flags.extend(gencode_flags)

if cuda_major >= 12:
    nvcc_flags.append('--allow-unsupported-compiler')
elif cuda_major == 11 and cuda_minor >= 8:
    pass # need anything?

# Define the extension
ext_modules = [
    cpp_extension.CUDAExtension(
        name='cuPRM',
        sources=[
            'src/bindings/py_bind.cpp',
            'src/collision/cc_2D.cu',
            'src/collision/env_2D.cu',
            'src/planning/construction.cu',
            'src/planning/pprm.cu',
            'src/local_planning/reedsshepp.cu',
        ],
        include_dirs=[
            'src/',
            pybind11.get_include(),
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-std=c++17',
                '-DTORCH_EXTENSION_NAME=cuPRM',
                '-DWITH_CUDA',
                '-fPIC'
            ],
            'nvcc': nvcc_flags
        },
        libraries=['cudart', 'curand'],  
        language='c++',
    )
]

setup(
    name='cuPRM',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
)