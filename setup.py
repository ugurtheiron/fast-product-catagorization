from pathlib import Path
from setuptools import find_packages, setup

# ---------------------------------------------------------------------------
# Helper: long description from README.md
# ---------------------------------------------------------------------------
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
setup(
    name="merlin",
    version="0.1.0",
    description="Merlin - Semantic e-commerce product classifier using OpenAI embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="<your-name>",
    author_email="<your-email>",
    url="<github-url>",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests*", "docs*", "examples*")),
    install_requires=[
        "openai>=1.0.0",
        "pandas>=2.2.0",
        "numpy>=1.25.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "ruff>=0.4.6", "wheel"],
        "docs": ["mkdocs-material"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # Enables:  merlin-classify "Product name"
            "merlin-classify = merlin.core:main",
        ]
    },
)