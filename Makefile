CONDA_ENV_NAME=qa_rag_env
PYTHON_VERSION=3.9
ENV_YML=environment.yml

# Third-party repositories to clone and install
# Format: REPO_NAME|REPO_URL
REPO_TIERS=\
	Pleias-Rag\|https://github.com/Pleias/Pleias-Rag.git \
#	AutreModule|https://github.com/exemple/AutreModule.git

.PHONY: all env clone shell update-env

all: env clone

env:
	@echo "🔍 Check Conda environment..."
	@if conda info --envs | grep -q "^$(CONDA_ENV_NAME)[[:space:]]"; then \
		echo "✅ The env '$(CONDA_ENV_NAME)' already exists."; \
	else \
		echo "⏳ Create env '$(CONDA_ENV_NAME)' from $(ENV_YML)..."; \
		conda env create -n $(CONDA_ENV_NAME) -f $(ENV_YML); \
	fi
	@echo "🧪 Check Python version..."
	@conda run -n $(CONDA_ENV_NAME) python --version | grep -q "^Python $(PYTHON_VERSION)" || \
		echo "⚠️ Attention : l'environnement n'est pas en Python $(PYTHON_VERSION)"

clone:
	@echo "📁 third-party dependecies install..."
	@mkdir -p src
	@for entry in $(REPO_TIERS); do \
		name=$$(echo $$entry | cut -d'|' -f1); \
		url=$$(echo $$entry | cut -d'|' -f2); \
		if [ -d "src/$$name" ]; then \
			echo "✅ Repository '$$name' already exists in src/."; \
		else \
			echo "⬇️ Clone '$$name' from $$url..."; \
			git clone $$url src/$$name; \
			echo "📦 Install '$$name' from conda env..."; \
			conda run -n $(CONDA_ENV_NAME) pip install src/$$name; \
		fi; \
	done

shell:
	@echo "💻 Open conda env in new shell ‘conda activate $(CONDA_ENV_NAME)’ (manual)"
	@conda run -n $(CONDA_ENV_NAME) bash

update-env:
	@echo "📝 Export env '$(CONDA_ENV_NAME)' to $(ENV_YML)..."
	@conda run -n $(CONDA_ENV_NAME) conda env export --from-history > $(ENV_YML)
	@echo "✅ $(ENV_YML) update manualy install dependecies (only!)."