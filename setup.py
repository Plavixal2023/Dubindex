from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dubindex",
    version="1.0.0",
    author="Jeff Hannel",
    author_email="support@dubindex.ai",
    description="Global LLM Ranking System using the DubIndex Formula",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Plavixal2023/dubindex",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "redis>=5.0.1",
        "pydantic>=2.4.2",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "schedule>=1.2.0",
        "huggingface-hub>=0.19.0",
        "PyGithub>=2.1.1"
    ],
    entry_points={
        "console_scripts": [
            "dubindex-api=dubindex.api:main",
        ],
    },
)
