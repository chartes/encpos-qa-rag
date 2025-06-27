CONDA_ENV_NAME=qa_rag_env
PYTHON_VERSION=3.9
ENV_YML=environment.yml

.PHONY: all env shell update-env clear-all

all: env clone

env:
	@echo "🔍 Check Conda environment..."
	@if conda info --envs | grep -q "^$(CONDA_ENV_NAME)[[:space:]]"; then \
		echo "✅ The env '$(CONDA_ENV_NAME)' already exists."; \
	else \
		echo "⏳ Create env '$(CONDA_ENV_NAME)'..."; \
		conda create -f $(ENV_YML) -y; \
	fi
	@echo "🧪 Check Python version..."
	@conda run -n $(CONDA_ENV_NAME) python --version | grep -q "^Python $(PYTHON_VERSION)" || \
		echo "⚠️ Attention : l'environnement n'est pas en Python $(PYTHON_VERSION)"

# conda create -y -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION); \


shell:
	@echo "💻 Open conda env in new shell ‘conda activate $(CONDA_ENV_NAME)’ (manual)"
	@conda run -n $(CONDA_ENV_NAME) bash

update-env:
	@echo "📝 Export env '$(CONDA_ENV_NAME)' to $(ENV_YML)..."
	@conda run -n $(CONDA_ENV_NAME) conda env export --from-history > $(ENV_YML)
	@echo "✅ $(ENV_YML) update manualy install dependecies (only!)."

clear-all:
	@echo "🗑️ Clear all conda env '$(CONDA_ENV_NAME)'..."
	@conda remove -y -n $(CONDA_ENV_NAME) --all
	@echo "🗑️ Clear all src/ directory..."
	@rm -rf src/*
	@echo "✅ All cleared."