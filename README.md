# encpos-qa-rag

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Conda](https://img.shields.io/badge/conda-available-green.svg)](https://docs.conda.io/en/latest/)

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)


This repository contains notebooks, code, application and ressources for the RAG LLM pipeline experiments
for ["Postions de thèses"](https://theses.chartes.psl.eu/) corpora de l'École nationale des chartes.

### Installation

- Clone the repository:
```bash
git clone https://github.com/chartes/encpos-qa-rag.git
cd encpos-qa-rag/
```

#### Option A: Using makefile
```bash
make
```
#### Option B: Manual installation

- create a conda environment:
```
conda env create -f environment.yml
```

- activate the environment:
```bash
conda activate qa_rag_env
```

- install requirements:
```bash
pip3 install -r requirements.txt
```

- First start by download [retrievers.zip]()
- Unzip the file in the `data/` directory


### [Notebooks](./notebooks) 

> [!TIP] 
> [`config.yml`](./config.yml) contains the configuration for the notebooks, including the paths to the data and the retrievers. You can modify it to suit your needs.

> [!WARNING] 
> Some data are already calculated and stored in the `data/` directory, you can use them directly without re-running the notebooks.

| Fichier                                                                    | Description                                                                                  |
|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`01-prepare_chunk_corpus.ipynb`](notebooks/01-prepare_chunk_corpus.ipynb) | Data analysis, preprocessing and chunking of the corpus of longform abstracts                |
| [`02-create_retrievers.ipynb`](notebooks/02-create_retrievers.ipynb)       | Vectorstore database creation (Retriever)                                                    |
| [`03-assemble_rag.ipynb`](notebooks/03-create_qa_rag.ipynb)                | Assemble the RAG pipeline with the retriever and the reader (generation part with LLM model) |


### [Streamlit application](./app/)

Check a specific documentation for [streamlit application](app/README.md)

```
@inproceedings{terjo2025from,
  title     = {From questions to insights: a reproducible question-answering pipeline for historiographical corpus exploration},
  author    = {Lucas Terriel and Vincent Jolivet},
  booktitle = {Proceedings of the Digital Humanities Conference (DH2025)},
  year      = {2025},
  address   = {Lisbon, Portugal},
  month     = {July 14-18},
  institution = {École nationale des chartes – PSL, France},
  note      = {Presented at DH2025, NOVA-FCSH}
}
```