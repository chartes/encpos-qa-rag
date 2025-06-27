# -*- coding: utf-8 -*-
"""vector_store_utils.py


"""

import os
import uuid
import shutil
import pandas as pd
from typing import Optional, List, Union
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(
        self,
        backend: str = 'faiss',  # 'faiss' or 'qdrant'
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 0,
        use_metadata: bool = True,
        metric: str = 'cosine',
        path: str = './vector_db',
        k: int = 5,
        qdrant_collection_name: str = 'my_collection',
        load_existing: bool = False,
        embedding_batch_size: int = 64
    ):
        self.backend = backend
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_metadata = use_metadata
        self.metric = metric
        self.path = path
        self.k = k
        self.embedding_batch_size = embedding_batch_size
        self.qdrant_collection_name = qdrant_collection_name

        # Text splitter if chunking enabled
        self.text_splitter = None
        if self.chunk_size:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        # Embedding model
        if backend == "faiss":
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"}
            )
        else:  # qdrant
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.index = None
        if backend == "faiss":
            self._init_faiss(load_existing)
        elif backend == "qdrant":
            self._init_qdrant(load_existing)

    def _init_faiss(self, load_existing):
        if load_existing and os.path.exists(self.path):
            self.index = LangchainFAISS.load_local(
                folder_path=self.path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )

    def _init_qdrant(self, load_existing):
        self.client = QdrantClient(path=self.path)
        distance = Distance.COSINE if self.metric == 'cosine' else Distance.EUCLID
        if not load_existing:
            self.client.recreate_collection(
                collection_name=self.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=distance
                )
            )

    def _split_text(self, text: str) -> List[str]:
        if self.text_splitter:
            return self.text_splitter.split_text(text)
        return [text]

    def add_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 64
    ):
        if self.backend == "faiss":
            self._add_to_faiss(df, text_column, metadata_columns)
        elif self.backend == "qdrant":
            self._add_to_qdrant(df, text_column, metadata_columns, batch_size)

    def _add_to_faiss(self, df, text_column, metadata_columns):
        documents = []

        for _, row in df.iterrows():
            texts = self._split_text(row[text_column])
            metadata = {col: row[col] for col in (metadata_columns or df.columns.drop(text_column))} if self.use_metadata else {}

            for chunk in texts:
                documents.append(Document(page_content=chunk, metadata=metadata))

        if not documents:
            return

        if self.index is not None:
            self.index.add_documents(documents)
        else:
            self.index = LangchainFAISS.from_documents(documents, self.embedding_model)

    def _add_to_qdrant(self, df, text_column, metadata_columns, batch_size):
        texts, payloads = [], []

        for _, row in df.iterrows():
            chunks = self._split_text(row[text_column])
            base_payload = {col: row[col] for col in (metadata_columns or df.columns.drop(text_column))} if self.use_metadata else {}

            for chunk in chunks:
                texts.append(chunk)
                payloads.append(base_payload)

        vectors = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True
        )

        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload=batch_payloads[j]
                )
                for j, vec in enumerate(batch_vectors)
            ]
            self.client.upsert(
                collection_name=self.qdrant_collection_name,
                points=batch_points,
                wait=True
            )

    def query(self, query_text: str, k: Optional[int] = None, threshold: Optional[float] = None) -> List:
        k = k or self.k

        if self.backend == "faiss":
            results = self.index.similarity_search(query_text, k=k)
            if threshold is not None:
                # FAISS returns similarity as relevance score, simulate threshold manually
                results = [r for r in results if getattr(r, "score", 1.0) >= threshold]
            return results

        elif self.backend == "qdrant":
            query_vector = self.embedding_model.encode(query_text)
            results = self.client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_vector,
                limit=k
            )
            filtered = [(r.id, r.score, r.payload) for r in results]

            if threshold is not None:
                if self.metric == 'cosine':
                    filtered = [r for r in filtered if r[1] >= threshold]
                elif self.metric == 'euclidean':
                    filtered = [r for r in filtered if r[1] <= threshold]

            return filtered

    def save(self):
        if self.backend == "faiss" and self.index is not None:
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            self.index.save_local(self.path)
