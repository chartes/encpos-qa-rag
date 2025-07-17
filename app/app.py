import os
import sys
import glob
import asyncio
from pathlib import Path

import streamlit as st


sys.path.append(str(Path(__file__).resolve().parent.parent))

from lib.rag_pipeline import RAGPipeline


st.set_page_config(
    layout="wide",
    page_title="Positions de thèses Assistant - test zone",
    page_icon="📜"
)

# === Vector DB registry ===
vector_db = {
    "camembert-base": {
        "model": "Lajavaness/sentence-camembert-base",
        "faiss": "../data/retrievers/vectordb/camembert-base_faiss",
        "lancedb": "../data/retrievers/vectordb/camembert-base_lancedb",
    },
    "camembert-large": {
        "model": "Lajavaness/sentence-camembert-large",
        "faiss": "../data/retrievers/vectordb/camembert-large_faiss",
        "lancedb": "../data/retrievers/vectordb/camembert-large_lancedb",
    },
    "multilingual": {
        "model": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "faiss": "../data/retrievers/vectordb/multilingual_faiss",
        "lancedb": "../data/retrievers/vectordb/multilingual_lancedb",
    },
}

@st.cache_resource(show_spinner="🔧 pipeline loading...")
def get_pipeline(
    question: str,
    retriever_only: bool,
    llm_model: str,
    template_path: str,
    backend: str,
    vectordb_path: str,
    embedding_model: str,
    bm25_path: str,
    k: int,
    hybrid: bool,
    rerank: bool,
) -> RAGPipeline:
    return RAGPipeline(
        query=question,
        retrieve_only=retriever_only,
        llm_model=llm_model,
        template_path=template_path,
        backend=backend,
        vectordb_path=vectordb_path,
        embedding_model=embedding_model,
        bm25_path=bm25_path,
        k=k,
        hybrid=hybrid,
        rerank=rerank,
        use_notebook=False,
        use_streaming=True,
    )



with st.sidebar:
    st.title("🛠️ Parameters")

    # LLM
    st.subheader("🔮 Language model section")
    #temperature = st.slider("Température", 0.0, 1.0, 0.0, 0.1)
    llm_model = st.text_input("LLM model name", "mistral-nemo-instruct-2407")

    # Retrieval
    st.subheader("📚 Retriever section")
    retriever_only = st.checkbox("Only retriever (no generation)")
    top_k = st.slider("Top-k documents", 3, 10, 5)
    backend = st.selectbox("Backend", ["faiss", "lancedb", "bm25"])

    embedding_model = None
    rerank = False
    hybrid_search = False

    if backend != "bm25":
        embedding_key = st.selectbox(
            "Embedding model",
            list(vector_db.keys()),
            format_func=lambda x: vector_db[x]["model"]
        )
        embedding_model = vector_db[embedding_key]["model"]
        rerank = st.checkbox("Reranker (CrossEncoder)")
        hybrid_search = st.checkbox("Hybrid Search (BM25 + vector)")

    # Template
    st.subheader("📜 Prompt section")
    templates = glob.glob("../prompt_templates/*.jinja")
    template_path = st.selectbox("Template Jinja", templates)
    template_content = ""
    if template_path:
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()
            st.text_area("template text", value=template_content, height=200)
        except Exception as e:
            st.error(f"Failed to load template : {e}")
            template_path = None


st.title("🎓 Research Assistant for Positions de thèses")
with st.form("question_form"):
    question = st.text_input("Ask a question")
    submit_button = st.form_submit_button("🚀 Send")


col_left, col_right = st.columns([2, 1])
response_container = col_left.empty()
doc_container = col_right.expander("📄 Retrieved documents and sections", expanded=True)
#doc_container.markdown("🔍 Aucun document affiché pour le moment...")


if submit_button and question and template_path:
    bm25_path = "../data/retrievers/bm25/bm25.encpos.tok.512_51.pkl"

    if backend == "bm25":
        embedding_key = "bm25"
        embedding_model = None
        vectordb_path = None
    else:
        vectordb_path = vector_db[embedding_key][backend]

    pipeline = get_pipeline(
        question=question,
        retriever_only=retriever_only,
        llm_model=llm_model,
        template_path=str(Path(template_path).resolve()),
        backend=backend,
        vectordb_path=vectordb_path,
        embedding_model=embedding_model,
        bm25_path=bm25_path,
        k=top_k,
        hybrid=hybrid_search,
        rerank=rerank,
    )

    async def run_pipeline():

        prompt = await pipeline.generate(question)
        if not prompt or prompt == "no documents found":
            st.warning("⚠️ Aucun prompt généré.")
            return
        docs = pipeline.relevant_docs


        if not retriever_only:
            with col_left:
                st.markdown("### 🤖 LLM Answer")
                container = st.empty()
                response = ""
                async for chunk in pipeline.llm.astream(pipeline.formatted_prompt):
                    token = getattr(chunk, "content", str(chunk))
                    response += token or ""
                    container.markdown(response + "▌")
                container.markdown(response + "\n\n" + pipeline.conclusion)

        grouped = {}
        for doc, score in docs:
            meta = doc.metadata
            key = (meta.get("author", "?"), meta.get("position_name", "?"), meta.get("year", "?"), meta.get("file_id", "?"))
            grouped.setdefault(key, []).append((doc, score))

        doc_container.empty()

        for (author, position_name, year, file_id), chunks in grouped.items():
            doc_container.markdown(f"**{author}**, *{position_name}*, promotion {year}")
            doc_container.markdown(
                f"\t🔗[Lien vers la position de thèse](https://theses.chartes.psl.eu/document/{file_id})")
            for doc, score in chunks:
                chunk_id = doc.metadata.get("chunk_id", "?")
                section = doc.metadata.get("section", "")
                doc_container.markdown(f"- Chunk {chunk_id} | **Score** : {score:.4f} | section : {section}")



        # display formatted prompt
        #with col_left:
        #    st.markdown("### 📜 Prompt utilisé")
        #    st.code(pipeline.formatted_prompt, language="jinja")



    with st.spinner("🔄 processing..."):
        asyncio.run(run_pipeline())