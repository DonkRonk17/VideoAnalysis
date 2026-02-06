#!/usr/bin/env python3
"""
VideoAnalysis - Setup Script
Enable AI agents to "watch" and analyze video content.

Built by ATLAS for Team Brain
Protocol: BUILD_PROTOCOL_V1.md
Requested by: WSL_CLIO (2026-02-04)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="videoanalysis",
    version="1.0.0",
    author="ATLAS (Team Brain)",
    author_email="logan@metaphysicsandcomputing.com",
    description="Enable AI agents to watch and analyze video content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DonkRonk17/VideoAnalysis",
    
    py_modules=["videoanalysis"],
    python_requires=">=3.9",
    
    install_requires=[
        "Pillow>=10.0.0",
    ],
    
    extras_require={
        "full": [
            "opencv-python>=4.8.0",
            "pytesseract>=0.3.10",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "cli": [
            "rich>=13.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "videoanalysis=videoanalysis:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    keywords=[
        "video", "analysis", "ai", "frames", "ocr", "scene-detection",
        "movement", "delta", "team-brain", "automation"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/DonkRonk17/VideoAnalysis/issues",
        "Source": "https://github.com/DonkRonk17/VideoAnalysis",
        "Documentation": "https://github.com/DonkRonk17/VideoAnalysis#readme",
    },
)
