import os
import shutil
import uuid
import json
import torch
import pickle
import lancedb
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings.base import Embeddings
from transformers import RagPreTrainedModel


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )[0].tolist()

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        kwargs.pop("normalize_embeddings", None)
        kwargs.pop("convert_to_numpy", None)
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            **kwargs
        )

class VectorDB:
    def __init__(
        self,
        backend: str = 'faiss',  # 'faiss' or 'lancedb'
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        use_metadata: bool = True,
        metric: str = 'cosine',
        path: str = './vector_db',
        k: int = 5,
        embedding_batch_size: int = 64,
        force_rebuild: bool = False
    ):
        assert backend in ["faiss", "lancedb"], "Backend must be 'faiss' or 'lancedb'."
        self.backend = backend
        self.embedding_model_name = embedding_model
        self.use_metadata = use_metadata
        self.metric = metric
        self.path = path
        self.k = k
        self.embedding_batch_size = embedding_batch_size
        self.force_rebuild = force_rebuild
        self._add_lancedb = True


        if backend in ["faiss", "lancedb"]:
            self.embedding_model = SentenceTransformerEmbeddings(
                model_name=embedding_model,
                normalize=(metric == "cosine")
            )

        self.index = None
        if backend == "faiss":
            self._init_faiss()
        elif backend == "lancedb":
            self._init_lancedb()

    def _init_faiss(self):
        if os.path.exists(self.path) and not self.force_rebuild:
            print(f"\U0001F4C2 Loading existing FAISS index from {self.path}")
            self.index = LangchainFAISS.load_local(
                folder_path=self.path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            if os.path.exists(self.path):
                print(f"⚠️ Rebuilding FAISS index at {self.path}")
                shutil.rmtree(self.path)
            else:
                print(f"🆕 Creating new FAISS index")
            self.index = None

    def _init_lancedb(self):
        print(f"📦 LanceDB intialization on: {self.path}")
        os.makedirs(self.path, exist_ok=True)
        self.db = lancedb.connect(self.path)
        self.table_name = os.path.basename(str(self.path).rstrip("/"))

        if self.table_name in self.db.table_names():
            print("ℹ️ LanceDB table founded.")
            self._add_lancedb = False
            if self.force_rebuild:
                self._add_lancedb = True
                print(f"🗑️ Remove Lance DB table: '{self.table_name}'")
                self.db.drop_table(self.table_name)
            else:
                self.index = LanceDB(
                    embedding=self.embedding_model,
                    connection=self.db,
                    table_name=self.table_name,
                    distance=self.metric,
                )
        else:
            print(f"🆕 Table LanceDB '{self.table_name}' à créer lors de l'indexation.")
            #self.index = self.db.open_table(self.table_name)
            self.index = LanceDB(
                embedding=self.embedding_model,
                connection=self.db,
                table_name=self.table_name,
                distance=self.metric,
            )

    def add_from_dataframe(self, df: pd.DataFrame, text_column: str = 'text', metadata_columns: Optional[List[str]] = None):
        if self.backend == "faiss":
            self._add_to_faiss(df, text_column, metadata_columns)
        elif self.backend == "lancedb":
            if self._add_lancedb:
                self._add_to_lancedb(df, text_column, metadata_columns)

    def _build_documents(self, df, text_column, metadata_columns):
        return [
            Document(
                page_content=row[text_column],
                metadata={col: row[col] for col in (metadata_columns or df.columns.drop(text_column))}
                if self.use_metadata else {}
            )
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Prepare docs for vector db backend: {self.backend}")
        ]

    def _add_to_faiss(self, df, text_column, metadata_columns):
        if self.index is not None:
            print("ℹ️ FAISS index already loaded — skipping creation.")
            return

        documents = self._build_documents(df, text_column, metadata_columns)
        if not documents:
            return

        self.index = LangchainFAISS.from_documents(
            documents,
            self.embedding_model,
        )

    def _add_to_lancedb(self, df, text_column, metadata_columns):
        if self.index is not None:
            print("ℹ️ LanceDB index already loaded — skipping creation.")
            return

        documents = self._build_documents(df, text_column, metadata_columns)
        if not documents:
            return

        print(f"🧠 Create new LanceDB table: '{self.table_name}' from docs...")
        self.index = LanceDB.from_documents(
            documents,
            embedding=self.embedding_model,
            connection=self.db,
            table_name=self.table_name
        )

    def embed_query(self, query: str) -> np.ndarray:
        if self.backend == "faiss":
            return np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        else:
            return self.embedding_model.encode(
                [query],
                normalize_embeddings=(self.metric == "cosine"),
                convert_to_numpy=True
            )


    def query(self, query_text: str, k: Optional[int] = 5, filter: str = "", search_type: str="similarity") -> List[
        Tuple[Document, float]]:
        """
        Effectue une requête de similarité sur la base vectorielle.

        Retourne : une liste de tuples (Document, score)
        - Le score est une **valeur de similarité normalisée entre 0 et 1**, où **1 = document parfaitement similaire**.
        - Le score est calculé comme `1 / (1 + distance)` quel que soit le backend ou le type de métrique :
            • Pour les embeddings normalisés (cosine distance), cela produit une mesure de similarité utile.
            • Pour les distances L2, cela agit comme une forme de "score inversé", plus haut = plus proche.

        Ce comportement est homogène entre FAISS et LanceDB.

        """
        k = k or self.k
        raw_results = self.index.similarity_search_with_score(query_text, k=k)
        results = [(doc, 1 / (1 + distance)) for doc, distance in raw_results]
        return results

    def save(self):
        if self.backend == "faiss" and self.index is not None:
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            self.index.save_local(self.path)

    def as_ensemble_retriever(self, bm25_retriever, weights=[0.5, 0.5]):
        if self.backend in ["faiss", "lancedb"]:
            vector_retriever = self.index.as_retriever(search_kwargs={"k": self.k})
        else:
            raise ValueError(f"Unsupported backend for ensemble: {self.backend}")

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights,

        )

class BM25Manager:
    def __init__(self, bm25_path="bm25_index.pkl", k=None):
        self.bm25_path = bm25_path
        self.k = k
        self.retriever = None

    def build_from_documents(self, documents):
        if os.path.exists(self.bm25_path):
            print(f"⚠️ BM25 index already exists at {self.bm25_path}. Loading existing index.")
            self.retriever = self.load()
            return self.retriever
        else:
            print("🛠️ Building BM25 retriever...")
            self.retriever = BM25Retriever.from_documents(documents)
            if self.k:
                self.retriever.k = self.k
            self.save()
            return self.retriever

    def save(self):
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.retriever, f)

    def load(self):
        with open(self.bm25_path, "rb") as f:
            self.retriever = pickle.load(f)
        return self.retriever

    def get_retriever(self):
        return self.retriever