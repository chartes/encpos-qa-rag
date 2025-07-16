CONDA_ENV_NAME=qa_rag_env
PYTHON_VERSION=3.10
ENV_YML=environment.yml
ENV_PIP_REQUIREMENTS=requirements.txt

.PHONY: all create-conda-env install-pip-requirements download-retrievers-from-github unzip-retrievers

all: create-conda-env install-pip-requirements download-retrievers-from-github unzip-retrievers

create-conda-env:
	@echo "Create conda env '$(CONDA_ENV_NAME)' with Python $(PYTHON_VERSION)..."
	@conda create -y -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)

install-pip-requirements:
	@echo "Install pip requirements from $(ENV_PIP_REQUIREMENTS)..."
	@conda run -n $(CONDA_ENV_NAME) pip3 install -r $(ENV_PIP_REQUIREMENTS)

download-retrievers-from-github:
	@echo "Download retrievers from GitHub..."
	@curl -o data/retrievers ...

unzip-retrievers:
	@echo "Unzip retrievers..."
	@unzip -o data/retrievers.zip -d data/retrievers
