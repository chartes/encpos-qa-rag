import json
import os
import random
import asyncio
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.rag_pipeline import RAGPipeline

# Paramètres
JSON_PATH = "eval_100-encpos-gpt4o.json"
llm_models = {
    "IBM-granite3.2-8B": "granite3.2-8b",
    #"Google-gemma2-9B":"gemma-2-9b",
    #"Mistral-nemo-12B": "mistral-nemo-instruct-2407",
    #"Qwen3-14B": "qwen3-14b",
}
N_QUESTIONS = 12

# Charger les questions
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
data = random.sample(data, N_QUESTIONS)

# Résultats
results = []

for llm_name, llm_model in llm_models.items():
    print(f"🧠 Modèle : {llm_name}")
    pipeline = RAGPipeline(
        llm_model=llm_model,
        retrieve_only=False,
        use_streaming=False,
        use_notebook=True,
        template_path="../../prompt_templates/v2.jinja",
        vectordb_path="/Users/lucaterre/Documents/pro/Travail_courant/DEV/AI-ENC-Projects/on-github/encpos-qa-rag/notebooks/scripts/data/vectordb/camembert-base_lancedb",
        embedding_model="Lajavaness/sentence-camembert-base",
        backend="lancedb",
        hybrid=True,
        bm25_path="/Users/lucaterre/Documents/pro/Travail_courant/DEV/AI-ENC-Projects/on-github/encpos-qa-rag/data/vectordb/bm25/bm25.encpos.tok.512_51.pkl"
    )
    #pipeline.set_llm_model(llm_model)

    for item in tqdm(data):
        question = item["question"]
        reference = item["answer"]

        try:
            response = asyncio.run(pipeline.generate(question, use_conclusion=False))
            if pipeline.relevant_docs:
                context = pipeline.relevant_docs[0][0].metadata.get("raw_chunk", "")
            else:
                context = ""
            response_cleaned = " ".join(response.split("\n")).strip()

            results.append({
                "model": llm_name,
                "question": question,
                "reference": reference,
                "response": response_cleaned,
                "context": context
            })

        except Exception as e:
            print(f"❌ Erreur : {e}")

# DataFrame et export CSV
df = pd.DataFrame(results)
df.to_csv("eval_generation_12_granite3.2-8b.csv", index=False)

print("✅ Export terminé")

