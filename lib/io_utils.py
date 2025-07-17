import yaml
import os
from pathlib import Path

def read_yaml(file_path):
    """Read a YAML file and return its contents."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_absolute_path(relative_path):
    """Convert a relative path to an absolute path."""
    return Path(__file__).resolve().parent.parent / relative_path

if __name__ == "__main__":
    # Example usage
    yaml_data = read_yaml('../config.yml')
    print(yaml_data)

    qdrant__path = yaml_data['vectorstore']['qdrant']["index_path"]
    print(qdrant__path)
    abs_path = get_absolute_path(qdrant__path)
    print(abs_path)