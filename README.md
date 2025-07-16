# encpos-qa-rag

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Conda](https://img.shields.io/badge/conda-available-green.svg)](https://docs.conda.io/en/latest/)

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)


This repository all the notebooks, code, application and ressources (data) for the RAG LLM pipeline 
for "Postions de thèses" corpora de l'École nationale des chartes.

### installation

- Clone the repository:
```bash
git clone ...
cd encpos-qa-rag/
```

- run make file: 
```bash
make
```

or

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


Now you can run the notebooks in the `notebooks/` directory or the Streamlit app in the `app/` directory.


#### [`notebooks/`](./notebooks) 

=> le fichier config.yml contient les paramètres de chunking et de prétraitement des données

| Fichier                                                                    | Description                                                                                |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`01-prepare_chunk_corpus.ipynb`](notebooks/01-prepare_chunk_corpus.ipynb) | Préparation et analyse des données, stratégie chunking                                     |
| [`02-create_retrievers.ipynb`](notebooks/02-create_retrievers.ipynb)       | Création des différentes bases vectorielles (Retriever)                                    |
| [`03-assemble_rag.ipynb`](notebooks/03-create_qa_rag.ipynb)                | Création du Reader et du pipeline RAG (Retriever+LLM) pour les positions de thèses de l'ENC |


#### test app 

Check a specific documentation for [streamlit application](app/README.md)

#### tree directory 
```bash
|-- encpos-qa-rag
  |-- data/

### Citation 

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