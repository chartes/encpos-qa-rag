# encpos-qa-rag

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Conda](https://img.shields.io/badge/conda-available-green.svg)](https://docs.conda.io/en/latest/)

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

A Question-answering RAG (Retrieval-augmented generation) pipeline for positions de thèses de l'ENC (ENCPOS).


#### [`notebooks/`](./notebooks) 

| Fichier                                                                        | Description                                                                                            |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| [`01-chunking_vector_index.ipynb`](./notebooks/01-chunking_vector_index.ipynb) | Préparation et analyse des données, stratégie chunking, vectorisation, indexation et test du retriever |
| [`02-assemble_qa_rag.ipynb`](./notebooks/02-assemble_qa_rag.ipynb)                                                 | Pipeline RAG (Retriever+LLM) pour les positions de thèses de l'ENC                                     |


=> décrire les données
=> scraper via Dots 
=> prétraitement chunking par sections et avec paramtères de chunking voir le config.yml
=> via Dataiku
=> section max et section min en caractères et en tokens

-> installer et lancer

-> arborescence des dossiers 

-> ajouter schéma 

-> lien article CR DH2025

-> tester app avec dans dossier preuve de concept HuggingFace Spaces ? 


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