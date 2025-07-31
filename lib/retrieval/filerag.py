import argparse
import traceback
import os
import sys
import re
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
from pathlib import Path
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from dataclasses import dataclass
import uuid
import pickle
import datetime

faiss.omp_set_num_threads(os.cpu_count() // 2)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

ticker_list = None
with open("data/cik/cikmap.json") as file:
    ticker_list = set(json.load(file).keys())

class metadata:
    def __init__(self, file: str = None, underlying: str = None, time: datetime.date = None):
        self.file = file
        self.underlying = underlying
        self.time = time

    def __str__(self):
        return "{" + f"file: {self.file}, underlying: {self.underlying}, time: {self.time}" + "}"

@dataclass
class Chunk:
    text: str
    chunk_id: str
    metadata: metadata
    embeddings: List[int]

    def __str__(self):
        return self.text


class FileEmbedder:
    """Handles file chunking, embedding, and FAISS indexing"""
    def __init__(self, model_name: str = "haophancs/bge-m3-financial-matryoshka"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = self.model.tokenizer

        self.index = None
        self.parent_store: Dict[str, Chunk] = {}
        self.faiss_id_to_parent_id: List[str] = []

    def _tokenize_and_chunk(self, text: str, chunk_size: int, overlap_ratio: float = 0.3) -> List[str]:
        tabular_blocks = re.findall(r'<TABULAR>.*?</TABULAR>', text, re.DOTALL)
        placeholder_template = "---TABULAR_PLACEHOLDER_{}---"
        
        text_with_placeholders = re.sub(r'<TABULAR>.*?</TABULAR>', 
                                        lambda m: placeholder_template.format(tabular_blocks.index(m.group(0))), 
                                        text, flags=re.DOTALL)

        initial_splits = re.split(r'\n---+\n|\n(?=Item \d+\w*\.)', text_with_placeholders)
        
        semantic_units_text = []
        for split in initial_splits:
            paragraphs = split.split('\n\n')
            semantic_units_text.extend([p.strip() for p in paragraphs if p.strip()])
        
        final_units_text = []
        for unit_text in semantic_units_text:
            placeholder_matches = list(re.finditer(r'---TABULAR_PLACEHOLDER_(\d+)---', unit_text))
            if placeholder_matches:
                last_end = 0
                processed_unit = ""
                for match in placeholder_matches:
                    start, end = match.span()
                    index = int(match.group(1))
                    processed_unit += unit_text[last_end:start]
                    processed_unit += tabular_blocks[index]
                    last_end = end
                processed_unit += unit_text[last_end:]
                final_units_text.append(processed_unit)
            else:
                final_units_text.append(unit_text)

        unit_tokens_list = []
        for unit in final_units_text:
            tokens = self.tokenizer.encode(unit, add_special_tokens=False)
            if len(tokens) > chunk_size:
                for i in range(0, len(tokens), chunk_size):
                    unit_tokens_list.append(tokens[i:i + chunk_size])
            else:
                unit_tokens_list.append(tokens)

        final_chunks_text = []
        current_chunk_tokens = []
        
        if not unit_tokens_list:
            return []

        current_chunk_tokens.extend(unit_tokens_list[0])
        
        for i in range(1, len(unit_tokens_list)):
            next_unit_tokens = unit_tokens_list[i]
            
            if len(current_chunk_tokens) + len(next_unit_tokens) > chunk_size:
                final_chunks_text.append(self.tokenizer.decode(current_chunk_tokens, skip_special_tokens=True))
                
                overlap_token_count = int(chunk_size * overlap_ratio)
                
                tokens_for_new_chunk = current_chunk_tokens[-overlap_token_count:] + next_unit_tokens

                current_chunk_tokens = tokens_for_new_chunk[:chunk_size]
            else:
                current_chunk_tokens.extend(next_unit_tokens)
                
        if current_chunk_tokens:
            final_chunks_text.append(self.tokenizer.decode(current_chunk_tokens, skip_special_tokens=True))

        return final_chunks_text

    def build_hierarchical_index(self, files: List[Path], parent_chunk_size: int = 1024, sub_chunk_size: int = 128, batch_size: int = 64):
        """
        Replaces chunk_markdown and embed_chunks. This method processes all files,
        creates a hierarchical index of parent/sub-chunks, and embeds them efficiently.
        """
        all_sub_chunk_embeddings = []
        faiss_id_counter = 0
        
        print(f"Processing {len(files)} files for hierarchical indexing...")
        for file_path in files:
            try:
                text = file_path.read_text(encoding='utf-8')
                parent_chunks_text = self._tokenize_and_chunk(text, chunk_size=parent_chunk_size, overlap_ratio=0.1)

                for parent_text in parent_chunks_text:
                    parent_id = str(uuid.uuid4())
                    parent_chunk = Chunk(text=parent_text, chunk_id=parent_id, metadata=metadata(file=str(file_path), underlying=file_path.parent.parent.name, time=datetime.datetime.strptime(file_path.name[:10], "%Y-%m-%d")), embeddings=[])
                    self.parent_store[parent_id] = parent_chunk

                    sub_chunks_text = self._tokenize_and_chunk(parent_text, chunk_size=sub_chunk_size, overlap_ratio=0.7)
                    
                    if not sub_chunks_text: continue

                    
                    for _ in range(len(sub_chunks_text)):
                        parent_chunk.embeddings.append(faiss_id_counter)
                        self.faiss_id_to_parent_id.append(parent_id)
                        faiss_id_counter += 1
                    
                    self.parent_store[parent_id] = parent_chunk
                        

                    with torch.no_grad():
                        if self.device == "cuda":
                            embeddings = self.model.encode(sub_chunks_text, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
                        else:
                            pool = self.model.start_multi_process_pool()
                            embeddings = self.model.encode_multi_process(sub_chunks_text, pool=pool, batch_size=batch_size)
                            self.model.stop_multi_process_pool(pool)

                    all_sub_chunk_embeddings.append(embeddings)
                print(f"Processed file {file_path.name}")
            except Exception as e:
                print(f"Error processing file {file_path.name}: {e}")

        if not all_sub_chunk_embeddings:
            print("No chunks were created. Aborting index build.")
            return

        print("Concatenating all sub-chunk embeddings...")
        final_embeddings = np.vstack(all_sub_chunk_embeddings)

        if self.index is None:
            dim = final_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        
        faiss.normalize_L2(final_embeddings)
        self.index.add(final_embeddings)
        print(f"Hierarchical index built successfully with {self.index.ntotal} sub-chunk pointers.")

    def search(self, query: str, k: int = 20, filters: List = []) -> List[Chunk]:
        """
        Searches the index of sub-chunk "pointers" and returns the full parent chunks.
        """
        if self.index is None: return []
        
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        distances, faiss_ids = self.index.search(q_emb, k)
        
        retrieved_parent_ids = set()
        for faiss_id in faiss_ids[0]:
            if faiss_id != -1:
                parent_id = self.faiss_id_to_parent_id[faiss_id]
                retrieved_parent_ids.add(parent_id)
        
        return list(filter(lambda x: x if all([i(x) for i in filters]) else False ,[self.parent_store[pid] for pid in retrieved_parent_ids]))

    def save(self, embeddings_dir: str = "data/embeddings"):
        print("\nSaving index and data stores...")
        output_dir = Path(embeddings_dir)
        output_dir.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "faiss_sub_chunk.index"))
        with open(output_dir / "parent_store.pkl", "wb") as f:
            pickle.dump(self.parent_store, f)
        with open(output_dir / "faiss_id_to_parent_id.pkl", "wb") as f:
            pickle.dump(self.faiss_id_to_parent_id, f)

    def load(self, embeddings_dir: str = "data/embeddings"):
        embeddings_path = Path(embeddings_dir)
        
        faiss_index_path = embeddings_path / "faiss_sub_chunk.index"
        parent_store_path = embeddings_path / "parent_store.pkl"
        faiss_mapping_path = embeddings_path / "faiss_id_to_parent_id.pkl"
        
        if not all(path.exists() for path in [faiss_index_path, parent_store_path, faiss_mapping_path]):
            raise FileNotFoundError("One or more embedding files not found. Please run build_hierarchical_index() first.")
        
        try:
            self.index = faiss.read_index(str(faiss_index_path))
            
            with open(parent_store_path, "rb") as f:
                self.parent_store = pickle.load(f)
            
            with open(faiss_mapping_path, "rb") as f:
                self.faiss_id_to_parent_id = pickle.load(f)
            
            print(f"Successfully loaded embeddings:")
            print(f"  - FAISS index with {self.index.ntotal} sub-chunk pointers")
            print(f"  - {len(self.parent_store)} parent chunks")
            print(f"  - {len(self.faiss_id_to_parent_id)} ID mappings")
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise

def mmr_subspace_rerank(query_embedding: np.ndarray, chunks: List[Chunk], faiss_index: faiss.Index, k: int = 20, lambda_mult: float = 0.7) -> List[Chunk]:
    """
    Performs MMR re-ranking using subspace similarity.
    It reconstructs sub-chunk embeddings from the main FAISS index.
    """
    if not chunks or query_embedding.size == 0:
        return []

    chunk_sub_embeddings = []
    for chunk in chunks:
        ids_to_reconstruct = np.array(chunk.embeddings, dtype='int64')
        if len(ids_to_reconstruct) > 0:
            reconstructed = faiss_index.reconstruct_batch(ids_to_reconstruct)
            chunk_sub_embeddings.append(reconstructed)
        else:
            chunk_sub_embeddings.append(np.array([]))

    relevance_scores = []
    for sub_embeddings in chunk_sub_embeddings:
        if sub_embeddings.shape[0] > 0:
            max_relevance = cosine_similarity(query_embedding, sub_embeddings).max()
            relevance_scores.append(max_relevance)
        else:
            relevance_scores.append(0)
    relevance_scores = np.array(relevance_scores)

    selected_indices = []
    remaining_indices = list(range(len(chunks)))

    if not remaining_indices: return []

    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < min(k, len(chunks)) and remaining_indices:
        best_score = -np.inf
        best_idx = -1

        for idx in remaining_indices:
            relevance_to_query = relevance_scores[idx]
            candidate_sub_embs = chunk_sub_embeddings[idx]
            if candidate_sub_embs.shape[0] == 0: continue

            max_similarity_to_selected = 0
            for sel_idx in selected_indices:
                selected_sub_embs = chunk_sub_embeddings[sel_idx]
                if selected_sub_embs.shape[0] == 0: continue
                
                similarity = cosine_similarity(candidate_sub_embs, selected_sub_embs).max()
                if similarity > max_similarity_to_selected:
                    max_similarity_to_selected = similarity

            mmr_score = lambda_mult * relevance_to_query - (1 - lambda_mult) * max_similarity_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [chunks[i] for i in selected_indices]

def rerank_with_bge_cross_encoder(
    query: str,
    chunks: List[Chunk],
    model_name: str = "BAAI/bge-reranker-base",
    k: int = 10,
    device: str = None
) -> List[Chunk]:
    """
    Reranks a list of Chunk objects using a BGE cross‑encoder and returns the top_k chunks.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(model_name, device=device)

    pair_list = [(query, chunk.text) for chunk in chunks]

    scores = reranker.predict(pair_list, show_progress_bar=False)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in scored_chunks[:k]]

    return top_chunks

summarizer = pipeline(
    "summarization",
    model="fahil2631/led-financial_summarization-genai15",
    tokenizer="fahil2631/led-financial_summarization-genai15",
    max_length=1024,
    min_length=50,
    do_sample=False
)

def summarize_each_chunk(query: str, context_chunks: List[Chunk], should_summarize: bool) -> List[Dict[str, str]]:
    prompt_template = """You are a financial analyst. Your task is to provide a clear and concise answer to the user's question based *only* on the provided context. Do not use any outside knowledge.

Follow these steps:
1.  First, carefully read the "Context" section to understand the information provided about the user's "Question".
2.  Identify all sentences and data points in the context that are directly relevant to the question.
3.  Synthesize these points into a coherent answer. If the context mentions specific impacts, numbers, or contributing factors, include them.
4.  If the context does not contain an answer, state that clearly.
5.  Keep numerical data if relevant or counterfactual.

Context:
{context}

Question: {query}

Answer:"""
    
    results = []
    
    print("Starting summarization...")
    for i, chunk in enumerate(context_chunks):
        try:
            if should_summarize:
                prompt = prompt_template.format(query=query, context=chunk.text)
                
                summary = summarizer(prompt)[0]["summary_text"]
            else:
                summary = chunk.text
            
            results.append({
                "summary": summary,
                "source_text": chunk.text,
                "chunk_id": chunk.chunk_id
            })
            
            print(f"Processed chunk {i+1}/{len(context_chunks)}")
            
        except Exception as e:
            print(f"Error summarizing chunk {chunk.chunk_id}: {e}")
            results.append({
                "summary": "[!] Error during summarization.",
                "source_text": chunk.text,
                "chunk_id": chunk.chunk_id
            })
    
    return results

def create_context(datadir: Path = Path(), embeddings_dir: Path = Path(), should_embed: bool = False) -> FileEmbedder:
    embedder = FileEmbedder()
    files = list(datadir.glob("**/*.md"))

    if should_embed:
        print(f"Creating embeddings")
        embedder.build_hierarchical_index(files)
        embedder.save(embeddings_dir)
        print(f"Saved embeddings in {embeddings_dir}")
    else:
        print(f"Loading embeddings from {embeddings_dir}")
        embedder.load(embeddings_dir)
    return embedder

def rag_search(context: FileEmbedder, search_q: str = "", should_summarize: bool = False) -> List[Dict[str, str]]:

    filters = []
    date = re.search(r"(?<!\d)20\d{2}(?!\d)", search_q, re.DOTALL)
    ticker = set(search_q.split(" ")).intersection(ticker_list)

    if date:
        filters.append(lambda x: str(x.metadata.time.year) == date.group())
    else:
        filters.append(lambda x: True)

    if ticker:
        filters.append(lambda x: x.metadata.underlying in ticker)
    else:
        filters.append(lambda x: True)

    retrieved_parent_chunks = context.search(search_q, k=100, filters=filters)

    query_embedding = context.model.encode([search_q])
    reranked_chunks = mmr_subspace_rerank(query_embedding, retrieved_parent_chunks, context.index, k=20)
    final_top_10 = rerank_with_bge_cross_encoder(search_q, reranked_chunks, k=10)

    return summarize_each_chunk(search_q, final_top_10, should_summarize)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Query SEC filings")
        parser.add_argument("query", help="The Query")
        parser.add_argument("-dd", "--data_dir", help="Directory the data comes from", default="data/filings")
        parser.add_argument("-ed", "--embeddings_dir", help="Directory embeddings come from", default="data/embeddings")
        parser.add_argument("-ss", "--summarize", help="Whether or not to create summaries", default="True")
        parser.add_argument("-se", "--embed", help="Whether or not to create embeddings", default="True")
        args = parser.parse_args()

        should_summarize = False if args.summarize.lower() == "false" else True
        should_embed = False if args.embed.lower() == "false" else True

        datadir = Path(args.data_dir)
        embeddings_dir = Path(args.embeddings_dir)
        search_q = args.query

        context = create_context(datadir=datadir, embeddings_dir=embeddings_dir, should_embed=should_embed)
        summaries = rag_search(context=context, search_q=search_q, should_summarize=should_summarize)

        print("\n=== Key Findings ===")
        for i, result in enumerate(summaries):
            print(f"[{i+1}] {result['summary']}")
            print(f"    Source File: {context.parent_store[result['chunk_id']].metadata.file}")
            print("---")

    except Exception:
        traceback.print_exc()
    finally:
        sys.exit()