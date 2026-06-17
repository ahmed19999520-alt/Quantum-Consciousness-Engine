
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-consciousness",
    version="0.1.0",
    author="Quantum Consciousness Team",
    author_email="dev@quantum-consciousness.ai",
    description="Quantum-Enhanced Consciousness AI Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-consciousness-ai/qc-library",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "qiskit>=0.36.0",
        "qiskit-aer>=0.10.0",
        "cirq>=0.14.0",
        "amazon-braket-sdk>=1.0.0",
        "pydantic>=1.8.0",
        "tqdm>=4.60.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900"
        ],
        "quantum": [
            "qiskit>=0.36.0",
            "qiskit-aer>=0.10.0",
            "cirq>=0.14.0",
            "amazon-braket-sdk>=1.0.0"
        ]
    }
)
