from typing import List, Tuple, Optional
import asyncio
import logging
from collections import defaultdict
import re
import os
import time

import lmstudio as lms
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from sentence_transformers import CrossEncoder

import numpy as np
from jinja2 import Template
from nltk.tokenize import sent_tokenize
import language_tool_python


from lib.vector_store import (VectorDB,
                                BM25Manager)


try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptBuilder:
    def __init__(
        self,
        template_path: str,
        max_total_tokens: int = 4096,
        output_buffer: int = 512,
        max_chunk_tokens: int = 500
    ):
        self.template_path = template_path
        self.max_total_tokens = max_total_tokens
        self.output_buffer = output_buffer
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = lms.llm()

    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def truncate_by_sentence(self, text: str, max_tokens: int) -> str:
        sentences = sent_tokenize(text)
        result = []
        for sent in sentences:
            result.append(sent)
            joined = " ".join(result)
            if self.get_token_count(joined) > max_tokens:
                result.pop()
                break
        return " ".join(result).strip() + " […]"

    def build_prompt(
        self,
        results: List[Tuple[Document, float]],
        question: str
    ) -> str:
        prompt_template = Template(open(self.template_path, encoding="utf-8").read())
        grouped = defaultdict(list)
        for doc, score in results:
            grouped[doc.metadata.get("file_id", "inconnu")].append((doc, score))


        sorted_groups = sorted(grouped.items(), key=lambda item: max(s for _, s in item[1]), reverse=True)
        included_chunks = []
        fallback_chunks = []
        token_budget = self.max_total_tokens - self.output_buffer

        header_base = prompt_template.render(context="PLACEHOLDER", question="PLACEHOLDER", annex="PLACEHOLDER")
        header_tokens = self.get_token_count(header_base.replace("{{context}}", "").replace("{{question}}", ""))
        total_tokens = header_tokens

        for file_id, chunks in sorted_groups:
            chunks.sort(key=lambda x: x[1], reverse=True)
            meta = chunks[0][0].metadata
            header = f"* Position de thèse : {meta.get('author','?')}, {meta.get('position_name','?')}, promotion {meta.get('year','?')}\n"
            section_lines = [header]

            for i, (doc, _) in enumerate(chunks):
                section = doc.metadata.get("section", "")
                extrait = doc.metadata.get("raw_chunk", doc.page_content).strip().replace("\n", " ")
                # extrait = truncate_by_sentence(extrait, max_chunk_tokens)

                line = f"Extrait {i+1}"
                if section:
                    line += f" - section « {section} »"
                line += f" : {extrait}"

                chunk_tokens = self.get_token_count(line)
                if total_tokens + chunk_tokens > token_budget:
                    fallback_chunks.append((meta, i + 1, section))
                    continue

                section_lines.append(line)
                total_tokens += chunk_tokens

            if len(section_lines) > 1:
                included_chunks.append("\n".join(section_lines) + "\n")

        context = "\n".join(included_chunks)

        annex = None
        if fallback_chunks:
            annex_by_thesis = defaultdict(list)
            for meta, i, section in fallback_chunks:
                file_id = meta.get("file_id", "inconnu")
                annex_by_thesis[file_id].append((meta, i, section))

            annex_lines = []
            for file_id, chunk_infos in annex_by_thesis.items():
                meta = chunk_infos[0][0]
                title = meta.get("position_name", "?")
                author = meta.get("author", "?")
                promo = meta.get("year", "?")
                header = f"* {author}, {title}, promotion {promo} :"
                annex_lines.append(header)

                for _, i, section in chunk_infos:
                    line = "\t- "
                    if section:
                        line += f"section « {section} »"
                    annex_lines.append(line)

            annex = "\n".join(annex_lines)

        try:
            return prompt_template.render(context=context, question=question, annex=annex)
        except:
            return prompt_template.render(context=context, question=question)

class RAGPipeline:
    def __init__(
        self,
        query: str = None,
        # LLM reader
        llm_model: str = "mistral-nemo-instruct-2407",
        template_path: str = "./prompt_templates/default.jinja",
        temperature: float = 0.0,
        base_url: str = "http://localhost:1234/v1",
        base_openai_key: str = "lm-studio",
        max_tokens: int = 4096,
        spell_check: bool = True,
        # Retriever
            retrieve_only: bool = False,
        backend: str = "faiss",
        vectordb_path: str = None,
        embedding_model: str = None,
        bm25_path: Optional[str] = None,
        k: int = 5,
        hybrid: bool = False,
        rerank: bool = False,
        reranker_model: str = "antoinelouis/colbertv1-camembert-base-mmarcoFR",
        # UI
        use_notebook: bool = True,
        use_streaming: bool = True
    ):
        self.query = query
        self.formatted_prompt = None
        self.relevant_docs = []
        self.retrieve_only = retrieve_only

        # Config
        self.use_notebook = use_notebook
        self.use_streaming = use_streaming
        self.max_tokens = max_tokens
        self.tool = language_tool_python.LanguageTool('fr')
        self.conclusion = (
            "Pour rappel, cette réponse est générée automatiquement à partir d'un modèle de langue. "
            "Elle peut contenir des approximations, des surcorrections, des erreurs factuelles ou des interprétations partielles. "
            "Il est vivement recommandé de vérifier les sources mentionnées et de consulter d'autres positions de thèses pour approfondir votre question."
        )

        # Prompt builder
        if not retrieve_only:
            self.prompt_builder = PromptBuilder(
            template_path=template_path,
            max_total_tokens=max_tokens,
            output_buffer=512,
            max_chunk_tokens=500
        )

        # Vector DB
        self.vector_db = None
        if vectordb_path and embedding_model:
            self.vector_db = VectorDB(
                backend=backend,
                path=vectordb_path,
                embedding_model=embedding_model,
                k=k
            )
        self.bm25 = None
        if backend == "bm25" or hybrid:
            if bm25_path:
                self.bm25 = BM25Manager(bm25_path=bm25_path, k=k).load()
        self.k = k
        self.hybrid = hybrid
        self.rerank = rerank
        self.reranker_model = reranker_model
        if self.rerank:
            self.reranker_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # LLM
        self.spell_check = spell_check
        self.llm = None
        self.llm_model = llm_model
        if not retrieve_only:
            self.llm = ChatOpenAI(
                model_name=self.llm_model,
                openai_api_base=base_url,
                openai_api_key=base_openai_key,
                streaming=use_streaming,
                max_tokens=max_tokens,
            )

    def set_llm_model(self, model_name: str):
        self.llm_model = model_name
        self.llm = ChatOpenAI(
            model_name=self.llm_model,
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="lm-studio",
            streaming=self.use_streaming,
            max_tokens=self.max_tokens,
        )

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        fetch_k = 20 if self.rerank else self.k

        if self.hybrid and self.bm25 and self.vector_db:
            logging.info("Using hybrid retriever (BM25 + vector)")
            retriever = self.vector_db.as_ensemble_retriever(self.bm25)
            docs = retriever.invoke(query)
            return [(doc, 1.0) for doc in docs[:fetch_k]]


        if self.vector_db:
            logging.info("Using vector retriever")
            return self.vector_db.query(query, k=fetch_k)

        if self.bm25 and not self.vector_db:
            logging.info("Using BM25 retriever")
            self.bm25.k = fetch_k
            docs = self.bm25.invoke(query)
            return [(doc, 1.0) for doc in docs]

        else:
            logging.warning("⚠️ No retriever.")
            return []

    def rerank_documents(self, docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        pairs = [(self.query, doc.page_content) for doc, _ in docs]
        scores = self.reranker_instance.predict(pairs)

        # Normalisation min-max pour [0,1]

        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            scores = [1.0] * len(scores)
        else:
            scores = [(score - min_score) / (max_score - min_score) for score in scores]

        # Normalization softmax :
        #exp_scores = np.exp(scores - np.max(scores))
        #scores = exp_scores / np.sum(exp_scores)

        # Sort by score
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [(doc, score) for (doc, _), score in reranked[:self.k]]

    def _mask_capitalized_words(self, text):
        masks = {}
        def replacer(match):
            key = f"__MASK{len(masks)}__"
            masks[key] = match.group(0)
            return key
        pattern = r'\b[A-ZÉÈÀÂÊÎÔÛÇ][a-zéèêàçîôûäëïöü]+\b'
        masked_text = re.sub(pattern, replacer, text)
        return masked_text, masks

    def _unmask_text(self, text, masks):
        for key, original in masks.items():
            text = text.replace(key, original)
        return text

    def _finalize_response(self, text: str, use_conclusion: bool=True) -> str:
        if not(use_conclusion):
            self.conclusion = ""

        lines = text.strip().split('\n')
        final_lines = []


        for line in lines:
            sentences = re.split(r'(?<=[.!?])\s+', line)
            if not sentences:
                continue
            if not re.search(r'[.!?]["”»”]?\s*$', sentences[-1]):
                sentences = sentences[:-1]
            if sentences:
                final_lines.append(" ".join(sentences))

        cleaned_text = "\n".join(final_lines).strip()
        if not cleaned_text:
            return self.conclusion

        if self.spell_check:
            masked_text, masks = self._mask_capitalized_words(cleaned_text)
            matches = self.tool.check(masked_text)
            corrected = language_tool_python.utils.correct(masked_text, matches)
            corrected_text = self._unmask_text(corrected, masks)
        else:
            corrected_text = cleaned_text

        return corrected_text + "\n\n" + self.conclusion

    async def _stream_and_print(self, prompt: str, timeout=100000) -> str:
        full_response = ""
        if self.use_notebook:
            output_display = display(Markdown("🟡 Processing..."), display_id=True)

        async def _run_stream():
            nonlocal full_response
            chunks = []
            async for chunk in self.llm.astream(prompt):
                text = getattr(chunk, "content", str(chunk))
                full_response += text
                chunks.append(text)
                if self.use_notebook:
                    output_display.update(Markdown("".join(chunks).replace("\n", "\n\n")))
                else:
                    print(text, end="", flush=True)

        try:
            await asyncio.wait_for(_run_stream(), timeout=timeout)
        except Exception as e:
            if self.use_notebook:
                output_display.update(Markdown(f"❌ **Erreur : {e}**"))
            else:
                print(f"❌ Erreur : {e}")
            return "Erreur"

        cleaned_response = self._finalize_response(full_response)
        if self.use_notebook:
            output_display.update(Markdown(cleaned_response.replace("\n", "\n\n")))
        else:
            print(cleaned_response)
        return cleaned_response

    async def generate_response(self, prompt: str, use_conclusion:bool=True) -> str:
        if self.use_streaming:
            return await self._stream_and_print(prompt)
        else:
            result = await self.llm.ainvoke(prompt)
            raw_response = getattr(result, "content", str(result))
            if self.use_notebook:
                display(Markdown("✅ **finished**"))
            else:
                print("\n✅ response OK")
            cleaned = self._finalize_response(raw_response, use_conclusion=use_conclusion)
            if self.use_notebook:
                display(Markdown(cleaned.replace("\n", "\n\n")))
            else:
                print(cleaned)
            return cleaned

    async def generate(self, query: str, use_conclusion:bool = True) -> str:
        self.query = query

        # === Step 1 : Retrieve documents ===
        start = time.time()
        self.relevant_docs = self.retrieve(query)
        end = time.time()
        logging.info(f"Retrieved {len(self.relevant_docs)} relevant documents in {end - start:.2f} seconds.")


        # === Step 2 : reranking (opt.) ===
        if self.rerank:
            logging.info("Reranking documents...")
            start = time.time()
            self.relevant_docs = self.rerank_documents(self.relevant_docs)
            end = time.time()
            logging.info(f"Reranked to {len(self.relevant_docs)} documents in {end - start:.2f} seconds.")

        # === Retrieval mode only ===
        if self.retrieve_only:
            logging.info("Retrieval only mode: skipping generation.")
            return "📄 Mode récupération uniquement activé. Aucune génération de réponse."

        # === Step 3: build the prompt ===
        start = time.time()

        self.formatted_prompt = self.prompt_builder.build_prompt(self.relevant_docs, query)
        if not self.formatted_prompt:
            message = "Je suis désolé, mais je ne peux pas répondre à votre question. Les positions de thèse que j'ai consultées ne contiennent pas d'informations sur ce sujet."
            logging.info(message)
            return message
        end = time.time()
        logging.info(f"Formatted prompt in {end - start:.2f} seconds.")

        if self.formatted_prompt == "no documents found":
            message = "Je suis désolé, mais je ne peux pas répondre à votre question. Les positions de thèse que j'ai consultées ne contiennent pas d'informations sur ce sujet."
            return message

        # === Step 4a: if notebook usage -> stream in good format
        if self.use_notebook:
            return await self.generate_response(self.formatted_prompt, use_conclusion=use_conclusion)

        # === Step 4b: no generate
        return self.formatted_prompt or "no documents found"