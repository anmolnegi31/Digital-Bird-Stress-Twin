"""
Setup script for Digital Bird Stress Twin package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="digital-bird-stress-twin",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade Digital Bird Stress Twin using ML/DL and MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/digital-bird-stress-twin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "torch>=2.0.1",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "librosa>=0.10.1",
        "mlflow>=2.5.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.3",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.4.1",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
    },
)
