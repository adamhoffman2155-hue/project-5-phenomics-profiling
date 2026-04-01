"""Setup configuration for the Phenomics Perturbation Profiling Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
req_file = Path(__file__).parent / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="phenomics-profiling",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description=(
        "Phenomics Perturbation Profiling Pipeline using OpenPhenom embeddings "
        "on RxRx3-core CRISPR and compound perturbation data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/phenomics-profiling",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.270",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phenomics-pipeline=scripts.run_pipeline:main",
            "phenomics-embed=scripts.embed_perturbations:main",
            "phenomics-cluster=scripts.cluster_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.py"],
    },
)
