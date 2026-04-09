import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_name = "src"

# Initilize the folder structure
list_of_folders = [
    ".github/workflow/.gitkeep",
    f"{project_name}/components/__init__.py",
    f"{project_name}/container/__init__.py",
    f"{project_name}/core/__init__.py",
    f"{project_name}/infrastructure/__init__.py",
    f"{project_name}/logs/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/interfaces/__init__.py",
    f"{project_name}/interfaces/llm/__init__.py",
    f"{project_name}/interfaces/chat/__init__.py",
    f"{project_name}/interfaces/infra/__init__.py",
    f"{project_name}/interfaces/logging/__init__.py",
    "requirement.txt",
    "setup.py",
    "ui/__init__.py",
    "backend/__init__.py",
    ".env",
    ".flake8",
]

for filepath in list_of_folders:
    path = Path(filepath)
    filedir = path.parent
    filename = path.name

    # Create directories if they don't exist
    if filedir != Path(""):
        os.makedirs(filedir, exist_ok=True)
        logging.info("Created Directory: %s", filedir)

    # Create files if they don't exist
    if not path.exists():
        path.touch()  # correctly call touch()
        logging.info("Created empty file: %s", filepath)
