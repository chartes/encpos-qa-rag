# Research Assistant for Positions de thèses

Streamlit Application to run the research assistant for positions de thèses
based on RAG framework.

## installation 

### 1. activate the conda environment

> to initialize the conda environment check main [README](../README.md)

```bash
conda activate qa_rag_env
```

### 2. (To use Generation part) Run [LMStudio](https://lmstudio.ai/) server 

> In LMStudio, download and serve the LLM `mistral-nemo-instruct-2407`.

### 2. Run the application

```bash
streamlit run app.py
```

### if fails 
```bash
conda install streamlit -c conda-forge
pip install --upgrade --force-reinstall streamlit
```
and again 
```bash
streamlit run app.py
```
