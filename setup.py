"""
Package configuration for MeaningMesh.
"""

from setuptools import setup, find_packages

setup(
    name="meaning_mesh",
    version="0.1.0",
    author="Mohammed Hamdan",
    description="A semantic text dispatching framework",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "huggingface": ["sentence-transformers>=2.0.0"],
        "cohere": ["cohere>=4.0.0"],
        "all": [
            "openai>=1.0.0",
            "sentence-transformers>=2.0.0",
            "cohere>=4.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
        ],
    },
)
