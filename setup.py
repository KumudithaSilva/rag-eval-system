from setuptools import find_packages, setup

"""

This script defines the package configuration for this Python project.
It is used by setuptools to install, package, and distribute the project.

Key responsibilities of setup.py:
- Defines metadata about the project (name, version, author, description, etc.)
- Specifies dependencies required to run the project
- Allows the project to be installed locally or via PyPI
- Supports creation of source distributions (.tar.gz) and binary wheels (.whl)
- Enables version management and structured project packaging

"""

setup(
    name="RAG-Eval-System",
    version="0.1.0",
    author="KumudithaSilva",
    author_email="kumudithasilva66@gmail.com",
    description="An open-source platform that able to benchmark rag techniques to identify the most accurate actionable insights.",
    long_description=open("README.md").read(),
    long_description_content_type="",
    maintainer="kumuditha",
    maintainer_email="kumudithasilva66@gmail.com",
    url="https://github.com/KumudithaSilva/rag-eval-system",
    project_urls={
        "Bug_Tracker": "https://github.com/KumudithaSilva/rag-eval-system/issues"
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10, <3.11",
)
