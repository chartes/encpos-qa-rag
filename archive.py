# ---------------------
# 1. INDEXER DANS QDRANT
# ---------------------

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import pandas as pd
import uuid
import numpy as np

# Load embedding model (français)
model = SentenceTransformer("Lajavaness/sentence-camembert-large")

# DataFrame avec textes + métadonnées
df = pd.DataFrame({
    "text": [
        "L'ADN est une molécule biologique essentielle.",
        "L'ARN transporte l'information génétique."
    ],
    "section": ["intro", "bases"],
    "type": ["scientifique", "scientifique"],
    "doc_id": [1, 2]
})

# Générer les vecteurs + ids
vectors = model.encode(df["text"].tolist())
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=vec.tolist(),
        payload={
            "text": row["text"],
            "section": row["section"],
            "type": row["type"],
            "doc_id": row["doc_id"]
        }
    ) for vec, (_, row) in zip(vectors, df.iterrows())
]

# Créer une collection Qdrant persistante (stockée sur disque)
client = QdrantClient(path="./qdrant_data")  # base locale persistée

client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=model.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    )
)

# Indexer les points
client.upsert(collection_name="documents", points=points)

print("✅ Indexation terminée dans Qdrant.")


# ---------------------
# 2. UTILISER EN MODE RAG AVEC PROMPT JINJA
# ---------------------

from qdrant_client.http import models as rest
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
import openai
from jinja2 import Template

# Config LLM local via LM Studio
openai.api_key = "lm-studio"
openai.api_base = "http://localhost:1234/v1"
llm = OpenAI(model="mistral-7b-instruct")

# Qdrant store pour LlamaIndex
vector_store = QdrantVectorStore(
    collection_name="documents",
    client=client,
    vector_name=""
)

# Recréer les documents pour LlamaIndex
docs = [
    Document(text=row["text"], metadata=row.to_dict()) for _, row in df.iterrows()
]

embedding = LangchainEmbedding(HuggingFaceEmbeddings(model_name="Lajavaness/sentence-camembert-large"))
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, embed_model=embedding, storage_context=storage_context)

# ---------------------
# PROMPT TEMPLATE AVEC JINJA
# ---------------------

prompt_template = Template("""
Tu es un expert scientifique. Voici les extraits de documents pertinents :

{{ context }}

Question : {{ question }}

Réponds de manière précise en français. Structure ta réponse au format JSON :
{
  "answer": "...",
  "source_ids": [...],
  "confidence": 0.xx
}
""")

# ---------------------
# REQUÊTE RAG AVEC COSINE THRESHOLD ET STREAMING
# ---------------------

from llama_index.core.retrievers import VectorIndexRetriever

query = "Qu'est-ce que l'ADN ?"
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
raw_results = retriever.retrieve(query)

# Appliquer un seuil de similarité cosine (par exemple >= 0.8)
COSINE_THRESHOLD = 0.8
filtered_results = [r for r in raw_results if r.score >= COSINE_THRESHOLD]

context = "\n".join([r.text for r in filtered_results])
prompt = prompt_template.render(context=context, question=query)

# Réponse streamée
response_stream = llm.stream_complete(prompt)
print("\nRéponse générée :\n")
streamed_text = ""
for token in response_stream:
    print(token.delta, end="", flush=True)
    streamed_text += token.delta

# Affichage des documents utilisés
print("\n\n📄 Documents utilisés :")
for r in filtered_results:
    meta = r.node.metadata
    print(f"\nDoc ID: {meta.get('doc_id')}")
    print(f"Section: {meta.get('section')}")
    print(f"Type: {meta.get('type')}")
    print(f"Texte: {meta.get('text')}")
