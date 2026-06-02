from setuptools import setup, find_packages

setup(
    name="wisdom",
    version="0.1.0",
    description="WISDOM: An explainable, white-box Coverage testing tool for DNNs",
    long_description=(
        "WISDOM is a PyTorch-based library for importance-driven deep neural networks "
        "system testing. It implements both the original IDC (DeepImportance) "
        "and an enhanced WISDOM pipeline with flexible attribution and clustering."
    ),
    author="Shenghao Qiu",
    license="Apache-2.0",
    packages = ["wisdom", "wisdom/core", "wisdom/attribution", "wisdom/clustering", "wisdom/utils", "wisdom/pruning"],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "torchaudio",
        "torchvision>=0.16",
        "scikit-learn",
        "numpy>=1.24",
        "matplotlib",
        "pytest",
        "setuptools",
        "onnx",
        "pyyaml",
        "opencv-python",
        "tqdm",
        "pandas",
        "botorch",
        "captum",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)