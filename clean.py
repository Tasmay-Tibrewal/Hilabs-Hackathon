#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HiLabs 2025 — Clinical Concept Harmonizer (Semantic + Fuzzy + LLM-Assist)

Adds:
- LLM-based query expansion (alternate keywords, description, likely STYs) via GPT-OSS-20B
- Weighted multi-signal scoring (desc/keywords/direct/STY)
- Large-K pipeline with fuzzy re-rank and per-code aggregation score
- Optional LLM re-ranking of final top codes with reasoning (XML)
- Keeps previous non-LLM pipeline switchable
"""

import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import re
import gc
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
import torch
from sentence_transformers import SentenceTransformer

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

from huggingface_hub import login as hf_login, hf_hub_download

import asyncio
import concurrent.futures
import multiprocessing
import uuid
import inspect
from datetime import datetime

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from contextlib import nullcontext

DATE_FORMAT_STR = "%Y-%m-%d %H:%M:%S"

def gpu_summary():
    if not torch.cuda.is_available():
        return 0, 0, 0
    gpus = torch.cuda.device_count()
    total = 0
    max_per = 0
    for i in range(gpus):
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, "total_memory", 0)
        total += mem
        max_per = max(max_per, mem)
    return gpus, total, max_per  # counts, sum bytes, max bytes


def default_concurrency():
    g, _, _ = gpu_summary()
    if g > 0:
        return 32 * g
    # CPU fallback
    cores = max(1, multiprocessing.cpu_count())
    return max(1, 2 * (cores - 1))

# -------------------- Constants & files --------------------
SNOMED = "SNOMEDCT_US"
RXNORM  = "RXNORM"

META_NAME          = "meta.json"
SNOMED_INDEX_NAME  = "snomed.index.faiss"
RXNORM_INDEX_NAME  = "rxnorm.index.faiss"
SNOMED_CAT_NAME    = "snomed_catalog.parquet"
RXNORM_CAT_NAME    = "rxnorm_catalog.parquet"
STY_VOCAB_JSON     = "sty_vocab.json"
STY_EMB_NPY        = "sty_embeddings.npy"

REQ_INPUT_COL = "Input Entity Description"
REQ_TYPE_COL  = "Entity Type"

OUT_SYS_COL   = "Output Coding System"
OUT_CODE_COL  = "Output Target Code"
OUT_DESC_COL  = "Output Target Description"
OUT_STY_COL   = "Output Semantic Type (STY)"
OUT_TTY_COL   = "Output Term Type (TTY)"

# ---------- Default Unsloth model choices ----------
# GGUF for llama.cpp (CPU or forced)
UNSLOTH_GGUF_QWEN_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
UNSLOTH_GGUF_QWEN_FILE = "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"

# vLLM (HF-style) repos – adjust if your org uses different names.
UNSLOTH_VLLM_OSS_20B = "Qwen/Qwen3-4B-Instruct-2507"  # GPT-OSS-20B HF weights
UNSLOTH_VLLM_QWEN4B  = "Qwen/Qwen3-4B-Instruct-2507"  # Qwen3 4B HF weights

# Default vLLM quantization to "bitsandbytes" (bnb)
DEFAULT_VLLM_QUANT = "bitsandbytes"
DEFAULT_VLLM_QUANT_QWEN4B = "bitsandbytes"

# -------------------- HF login --------------------
def maybe_hf_login(token: Optional[str]):
    if token:
        try:
            hf_login(token=token)
            print("[HF] Logged in to HuggingFace Hub.")
        except Exception as e:
            print(f"[HF] Login failed: {e}")

def pick_llm_backend_and_model(backend, hf_model_id, gguf_path):
    # Respect explicit backend choice first
    if backend in ("vllm", "llama-cpp"):
        if backend == "vllm":
            if gguf_path:
                print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Note: gguf_path is ignored by vLLM (GGUF is llama.cpp-only).")
            # default Unsloth vLLM target if none supplied
            if not hf_model_id or hf_model_id in ("auto", ""):
                hf_model_id = UNSLOTH_VLLM_OSS_20B
        else:  # llama-cpp
            if not gguf_path or gguf_path in ("auto", ""):
                gguf_path = hf_hub_download(
                    repo_id=UNSLOTH_GGUF_QWEN_REPO, filename=UNSLOTH_GGUF_QWEN_FILE
                )
        return backend, hf_model_id, gguf_path

    # backend == auto
    g, total_bytes, max_bytes = gpu_summary()
    max_gb = max_bytes / (1024**3)

    if g == 0:
        # CPU → llama.cpp + Qwen3 4B GGUF
        if not gguf_path:
            gguf_path = hf_hub_download(
                repo_id=UNSLOTH_GGUF_QWEN_REPO, filename=UNSLOTH_GGUF_QWEN_FILE
            )
        return "llama-cpp", None, gguf_path

    # GPU present → vLLM
    # If max per-GPU VRAM < 22 GB → prefer Qwen3 4B HF (bnb)
    if max_gb < 22.0:
        return "vllm", (hf_model_id or UNSLOTH_VLLM_QWEN4B), None

    # Otherwise prefer GPT-OSS-20B HF (bnb)
    return "vllm", (hf_model_id or UNSLOTH_VLLM_OSS_20B), None

# -------------------- CUDA memory helpers --------------------
def clear_cuda_memory(note: str = ""):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    if note:
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**3)
                reserv = torch.cuda.memory_reserved() / (1024**3)
                print(f"[CUDA] Cleared {note} | allocated={alloc:.2f} GB, reserved={reserv:.2f} GB")
            else:
                print(f"[CUDA] Cleared {note}")
        except Exception:
            print(f"[CUDA] Cleared {note}")

# >>> UPDATE START
def configure_cpu_threading(blas_threads: Optional[int] = None,
                            faiss_threads: Optional[int] = None):
    """
    - Sets FAISS OpenMP threads (affects `index.search` etc.)
    - Caps/sets BLAS threadpools for NumPy/Scipy (OpenBLAS/MKL/BLIS/NumExpr)
    """
    if faiss_threads:
        try:
            faiss.omp_set_num_threads(int(faiss_threads))
            print(f"[Threads] FAISS OMP threads = {int(faiss_threads)}")
        except Exception as e:
            print(f"[Threads] Failed to set FAISS threads: {e}")

    if blas_threads:
        n = int(blas_threads)
        # env vars for common BLAS backends (best-effort)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(n)
        if threadpool_limits:
            try:
                # Note: threadpool_limits is a context manager; for "global" behavior,
                # we’ll wrap our heavy sections in a `with` block (see below).
                print(f"[Threads] BLAS thread cap requested = {n}")
            except Exception as e:
                print(f"[Threads] threadpool_limits not available: {e}")
# <<< UPDATE END

def model_roundtrip_cpu_gpu(model):
    if not torch.cuda.is_available():
        return model
    try:
        model.to("cpu")
    except Exception:
        pass
    clear_cuda_memory("after model->CPU")
    try:
        model.to("cuda")
    except Exception:
        pass
    clear_cuda_memory("after model->CUDA")
    return model

# -------------------- Embedding model --------------------
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device())

def embed_texts(model: SentenceTransformer,
                texts: List[str],
                batch_size: int = 1024,
                normalize: bool = True) -> np.ndarray:
    """OOM-resilient embedding with batch backoff."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    chunks: List[np.ndarray] = []
    i, bs = 0, max(1, int(batch_size))
    with torch.inference_mode():
        while i < len(texts):
            j = min(i + bs, len(texts))
            try:
                vecs = model.encode(
                    texts[i:j],
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
                if vecs.dtype != np.float32:
                    vecs = vecs.astype(np.float32, copy=False)
                chunks.append(vecs)
                i = j
            except torch.cuda.OutOfMemoryError:
                clear_cuda_memory("OOM during embed; backoff")
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
    return np.vstack(chunks)

# -------------------- Index builders --------------------
def build_hnsw(d: int, m: int = 32, efc: int = 200) -> faiss.Index:
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    faiss.downcast_index(index).hnsw.efConstruction = efc
    return index

def build_flat(d: int) -> faiss.Index:
    return faiss.IndexFlatIP(d)

def build_ivfpq(d: int, nlist: int = 16384, pq_m: int = 64, pq_nbits: int = 8) -> faiss.Index:
    quant = faiss.IndexFlatIP(d)
    return faiss.IndexIVFPQ(quant, d, nlist, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)

def choose_system(entity_type: str) -> str:
    return RXNORM if str(entity_type).strip().lower() == "medicine" else SNOMED

# -------------------- Catalog prep --------------------
def prepare_catalog(df: pd.DataFrame, system_name: str) -> pd.DataFrame:
    keep_cols = ["CUI", "System", "TTY", "CODE", "STR", "STY"]
    df = df.loc[:, keep_cols].copy()
    df["System"] = system_name
    df["CODE"] = df["CODE"].astype(str)
    df["STR"]  = df["STR"].astype(str)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(np.int64)
    return df[["row_id", "CODE", "STR", "CUI", "TTY", "STY", "System"]]

def train_ivfpq_if_needed(index: faiss.Index, vecs_for_training: np.ndarray):
    inner = index
    if isinstance(index, faiss.IndexIDMap2):
        inner = index.index
    if isinstance(inner, faiss.IndexIVFPQ):
        inner.train(vecs_for_training)

# -------------------- LLM backends (vLLM / llama.cpp) --------------------
# @dataclass
# class LLMConfig:
#     backend: str = "auto"  # auto|vllm|llama-cpp
#     hf_model_id: str = UNSLOTH_VLLM_OSS_20B  # for vLLM
#     gguf_path: Optional[str] = None  # for llama-cpp
#     max_new_tokens: int = 512
#     temperature: float = 0.1
#     top_p: float = 0.9

@dataclass
class LLMConfig:
    backend: str = "auto"      # auto|vllm|llama-cpp
    hf_model_id: Optional[str] = UNSLOTH_VLLM_OSS_20B
    gguf_path: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    vllm_quantization: Optional[str] = None
    tp: Optional[int] = None    # tensor parallel
    n_threads: Optional[int] = None
    concurrency: Optional[int] = None

class AsyncLLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.mode = None
        self.engine = None
        self.sampling = None
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = asyncio.Lock()  # safety for llama shared context
        self._limiter = asyncio.BoundedSemaphore(int(self.cfg.concurrency or default_concurrency()))
        self._init()

    def _auto_defaults(self):
        if self.cfg.concurrency is None:
            self.cfg.concurrency = default_concurrency()
        if self.cfg.tp in (None, "auto"):
            self.cfg.tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if self.cfg.tp < 1:
                self.cfg.tp = 1
        if self.cfg.n_threads in (None, "auto"):
            self.cfg.n_threads = max(1, multiprocessing.cpu_count() - 1)
    
    def _init(self):
        self._auto_defaults()
        backend, hf_id, gguf = pick_llm_backend_and_model(self.cfg.backend,
                                                          self.cfg.hf_model_id,
                                                          self.cfg.gguf_path)
        self.cfg.backend, self.cfg.hf_model_id, self.cfg.gguf_path = backend, hf_id, gguf

        if backend == "vllm":
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            # Hard-cap TP to visible GPUs
            tp = int(self.cfg.tp or 1)
            if torch.cuda.is_available():
                tp = max(1, min(tp, torch.cuda.device_count()))
            else:
                tp = 1

            # Default quant = bitsandbytes unless overridden
            quant = self.cfg.vllm_quantization
            if quant in (None, "", "auto"):
                is_qwen4b = (self.cfg.hf_model_id == UNSLOTH_VLLM_QWEN4B)
                quant = DEFAULT_VLLM_QUANT if is_qwen4b else DEFAULT_VLLM_QUANT_QWEN4B
            self.cfg.vllm_quantization = quant

            if self.cfg.gguf_path:
                print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Note: gguf_path is ignored on vLLM (use llama-cpp for GGUF).")

            if quant is not None:
                eng_args = AsyncEngineArgs(
                    model=self.cfg.hf_model_id or UNSLOTH_VLLM_OSS_20B,
                    dtype="auto",
                    trust_remote_code=True,
                    tensor_parallel_size=tp,
                    disable_log_stats=True,
                    quantization=quant,
                    gpu_memory_utilization=0.9,
                    max_num_seqs=2048,
                    enable_prefix_caching=True,          # huge win given identical system prompt
                    enforce_eager=True,                 # avoid lazy init stalls
                )
            
            else:
                eng_args = AsyncEngineArgs(
                    model=self.cfg.hf_model_id or UNSLOTH_VLLM_OSS_20B,
                    dtype="auto",
                    trust_remote_code=True,
                    tensor_parallel_size=tp,
                    disable_log_stats=True,
                    gpu_memory_utilization=0.9,
                    max_num_seqs=2048,
                    enable_prefix_caching=True,          # huge win given identical system prompt
                    enforce_eager=True,                 # avoid lazy init stalls
                )

            # Optional: hint for FlashInfer
            try:
                import flashinfer  # noqa: F401
                print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] FlashInfer detected.")
            except Exception:
                print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] FlashInfer not detected. For best perf: `pip install 'vllm[flashinfer]'`")

            self.engine = AsyncLLMEngine.from_engine_args(eng_args)
            self.sampling = SamplingParams(
                max_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
            self.mode = "vllm"
            print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] vLLM ready. model={eng_args.model} tp={tp} quant={quant}")
        else:
            # llama-cpp
            from llama_cpp import Llama
            self.engine = Llama(model_path=self.cfg.gguf_path,
                                n_ctx=131072,  # large ctx; adjust if needed
                                n_threads=int(self.cfg.n_threads),
                                logits_all=False,
                                verbose=False)
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=int(self.cfg.concurrency))
            self.mode = "llama-cpp"
            print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] llama-cpp ready. gguf={self.cfg.gguf_path} threads={self.cfg.n_threads} "
                  f"pool={self.cfg.concurrency}")

    async def generate_one(self, system_prompt: str, user_prompt: str) -> str:
        async with self._limiter:
            if self.mode == "vllm":
                prompt = (
                    f"<|system|>\n{system_prompt}\n</s>\n"
                    f"<|user|>\n{user_prompt}\n</s>\n"
                    f"<|assistant|>\n"
                )
                request_id = str(uuid.uuid4())

                nlc = '\n'
                req = self.engine.generate(prompt, self.sampling, request_id)

                # vLLM V1 returns an async generator (stream). Older wrappers may return an awaitable or a value.
                try:
                    if inspect.isasyncgen(req):
                        last = None
                        async for out in req:        # consume stream to completion
                            last = out
                        # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] recieved output for req {request_id}: {last.outputs[0].text.strip().replace(nlc,' ')[:100] if last and last.outputs else ''}")
                        return last.outputs[0].text if last and last.outputs else ""
                    elif inspect.isawaitable(req):
                        out = await req
                        # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] recieved output for req {request_id}: {out.outputs[0].text.strip().replace(nlc,' ')[:100] if out and out.outputs else ''}")
                        return out.outputs[0].text if out and out.outputs else ""
                    else:
                        out = req
                        # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] recieved output for req {request_id}: {out.outputs[0].text.strip().replace(nlc,' ')[:100] if out and out.outputs else ''}")
                        return out.outputs[0].text if out and out.outputs else ""
                except Exception:
                    print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] recieved exception for req {request_id}.")
                    # optional: log here
                    return ""
            else:
                # llama-cpp is blocking; run in thread pool
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                loop = asyncio.get_running_loop()
                async with self._lock:  # avoid unsafe concurrent access
                    return await loop.run_in_executor(
                        self.executor,
                        lambda: self.engine.create_chat_completion(
                            messages=msgs,
                            temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                            max_tokens=self.cfg.max_new_tokens,
                        )["choices"][0]["message"]["content"]
                    )

    async def generate_many(self, pairs: List[Tuple[str, str]]) -> List[str]:
        # throttle via semaphore
        return await asyncio.gather(*(self.generate_one(sp, up) for sp, up in pairs))

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.impl = None
        self.mode = None
        self.SamplingParams = None
        self._init()

    def _auto_defaults(self):
        if self.cfg.tp in (None, "auto"):
            self.cfg.tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if self.cfg.tp < 1:
                self.cfg.tp = 1
        if self.cfg.n_threads in (None, "auto"):
            self.cfg.n_threads = max(1, multiprocessing.cpu_count() - 1)
        if self.cfg.vllm_quantization in (None, "", "auto"):
            is_qwen4b = (self.cfg.hf_model_id == UNSLOTH_VLLM_QWEN4B)
            self.cfg.vllm_quantization = DEFAULT_VLLM_QUANT if not is_qwen4b else DEFAULT_VLLM_QUANT_QWEN4B

    def _init(self):
        backend, hf_id, gguf = pick_llm_backend_and_model(
            self.cfg.backend, self.cfg.hf_model_id, self.cfg.gguf_path
        )
        self.cfg.backend, self.cfg.hf_model_id, self.cfg.gguf_path = backend, hf_id, gguf

        self._auto_defaults()

        if backend == "vllm":
            from vllm import LLM as VLLM, SamplingParams
            tp = int(self.cfg.tp or 1)
            if torch.cuda.is_available():
                tp = max(1, min(tp, torch.cuda.device_count()))
            if self.cfg.gguf_path:
                print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Note: gguf_path is ignored on vLLM (use llama-cpp for GGUF).")
            self.SamplingParams = SamplingParams
            self.impl = VLLM(
                model=self.cfg.hf_model_id,
                trust_remote_code=True,
                tensor_parallel_size=tp,
                quantization=self.cfg.vllm_quantization,
            )
            self.mode = "vllm"
            print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Using vLLM. model={self.cfg.hf_model_id} tp={tp} quant={self.cfg.vllm_quantization}")
        else:
            # llama-cpp (CPU or forced)
            from llama_cpp import Llama
            if not self.cfg.gguf_path:
                self.cfg.gguf_path = hf_hub_download(
                    repo_id=UNSLOTH_GGUF_QWEN_REPO, filename=UNSLOTH_GGUF_QWEN_FILE
                )
            self.impl = Llama(
                model_path=self.cfg.gguf_path,
                n_ctx=131072,
                n_threads=int(self.cfg.n_threads),
                logits_all=False,
                verbose=False,
            )
            self.mode = "llama-cpp"
            print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Using llama-cpp. gguf={self.cfg.gguf_path} threads={self.cfg.n_threads}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.mode == "vllm":
            sp = self.SamplingParams(
                max_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
            prompt = f"<|system|>\n{system_prompt}\n</s>\n<|user|>\n{user_prompt}\n</s>\n<|assistant|>\n"
            outs = self.impl.generate(prompt, sp)
            return outs[0].outputs[0].text
        else:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            out = self.impl.create_chat_completion(
                messages=msgs,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_new_tokens,
            )
            return out["choices"][0]["message"]["content"]

# -------------------- LLM prompts & XML parsing --------------------
EXPAND_SYSTEM_PROMPT = """You are an expert clinical terminology normalizer. 
Task: normalize messy clinical inputs into standardized concepts for mapping to RxNorm (medications) and SNOMED CT (labs, diagnoses, procedures).
Inputs may be abbreviated, colloquial, misspelled, or non-standard (e.g., "paracetamol" vs "Acetaminophen", "xr chest" vs "X-ray chest").
Return results STRICTLY as XML with one or more <candidate> blocks.
Each <candidate> block describes a distinct possible interpretation of the input.
It is for the case, when you are not sure about the exact meaning, to cover multiple plausible interpretations.
It is optional to give multiple candidates, if you are sure about the meaning, a single candidate is fine.
But if you are unsure, give multiple candidates to cover the possibilities (ideally 2-3, but upto 5 if needed).
For each candidate, include:
  <alternate_keywords>comma-separated short terms and brand/common names along with scientific names (give 2-5, and at max 10 alternate keywords)</alternate_keywords>
  <description>clear medical explanation (1-3 sentences)</description>
  <possible_semantic_term_types>comma-separated choices FROM THE PROVIDED LIST ONLY</possible_semantic_term_types>
Do not add any other tags. Do not include markdown. Use only ASCII punctuation.
Semantic type choices (union of RxNorm & SNOMED CT types):
{sty_inventory}
"""

EXPAND_USER_PROMPT = """Input Entity Description: {text}
Entity Type: {entity_type}

Output the XML now.
"""

RERANK_SYSTEM_PROMPT = """You are ranking standardized clinical codes for a messy input. 
You receive the original input & entity type, plus up to 50 candidate codes (from RxNorm and SNOMED CT) with strings and types.
Goal: select the BEST top-N codes consistent with the input meaning and entity type.
Guidance:
- Prefer RxNorm for medications; prefer SNOMED CT for labs/diagnoses/procedures. 
- However, if preferred coding system is missing or clearly worse, it's OK to choose the other system.
- Consider code string (STR), term type (TTY), semantic type (STY), and overall plausibility.
Return STRICT XML: a sequence of <choice> blocks; each contains:
  <code>THE EXACT CODE STRING</code>
  <reasoning>concise justification</reasoning>
Do not include other tags, no markdown, no extra text.
"""

RERANK_USER_PROMPT = """Original Query: {text}
Entity Type: {entity_type}

Expanded understanding (from earlier step):
{expanded_summary}

Candidate codes (system | code | STY | TTY | one or more names):
{candidates_blob}

Return the XML with up to {final_k} <choice> items.
"""

def extract_xml_candidates(xml_text: str) -> List[Dict[str, Any]]:
    """
    Parse <candidate> blocks from expansion XML.
    Robust to minor formatting issues; falls back to regex if needed.
    """
    # Simple normalization
    s = xml_text.strip()
    # Greedy but robust: split by <candidate>...</candidate>
    blocks = re.findall(r"<candidate>(.*?)</candidate>", s, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        # Try single-block format without outer tags
        blocks = [s]
    results = []
    for b in blocks:
        def grab(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", b, flags=re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""
        kws = grab("alternate_keywords")
        desc = grab("description")
        stys = grab("possible_semantic_term_types")
        kw_list = [k.strip() for k in re.split(r"[,\n;]", kws) if k.strip()]
        sty_list = [k.strip() for k in re.split(r"[,\n;]", stys) if k.strip()]
        if not kw_list and not desc and not sty_list:
            continue
        results.append({
            "alternate_keywords": kw_list,
            "description": desc,
            "possible_stys": sty_list
        })
    if not results:
        # Fallback: treat entire text as one candidate description
        results = [{"alternate_keywords": [], "description": s[:400], "possible_stys": []}]
    return results

def extract_xml_choices(xml_text: str) -> List[Dict[str, str]]:
    """Parse <choice><code>…</code><reasoning>…</reasoning></choice>"""
    blocks = re.findall(r"<choice>(.*?)</choice>", xml_text, flags=re.DOTALL | re.IGNORECASE)
    out = []
    for b in blocks:
        code = re.search(r"<code>(.*?)</code>", b, flags=re.DOTALL | re.IGNORECASE)
        reas = re.search(r"<reasoning>(.*?)</reasoning>", b, flags=re.DOTALL | re.IGNORECASE)
        if code:
            out.append({"code": code.group(1).strip(), "reasoning": (reas.group(1).strip() if reas else "")})
    return out

async def llm_expand_query_async(llm: AsyncLLMClient, text: str, entity_type: str, sty_inventory: List[str]) -> List[Dict]:
    sys_prompt = EXPAND_SYSTEM_PROMPT.format(sty_inventory="\n".join(f"- {s}" for s in sty_inventory))
    user_prompt = EXPAND_USER_PROMPT.format(text=text, entity_type=entity_type)
    nlc = '\n'
    print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Expanding query for: {text[:60].replace(nlc, ' ')}...")
    out = await llm.generate_one(sys_prompt, user_prompt)
    return extract_xml_candidates(out)

async def llm_rerank_codes_async(llm: AsyncLLMClient, text: str, entity_type: str,
                                 expanded_summary: str, candidates_blob: str, final_k: int) -> List[str]:
    sys_prompt = RERANK_SYSTEM_PROMPT
    user_prompt = RERANK_USER_PROMPT.format(text=text, entity_type=entity_type,
                                            expanded_summary=expanded_summary,
                                            candidates_blob=candidates_blob,
                                            final_k=final_k)
    out = await llm.generate_one(sys_prompt, user_prompt)
    return [c["code"] for c in extract_xml_choices(out)][:final_k]

# -------------------- BUILD --------------------
def build_indices(args):
    maybe_hf_login(args.hf_token)

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"[Build] Embedding model: {args.model} (dim={dim}, device={device()})")

    snomed_df = pd.read_parquet(args.snomed_parquet)
    rxnorm_df = pd.read_parquet(args.rxnorm_parquet)

    snomed_cat = prepare_catalog(snomed_df, SNOMED)
    rxnorm_cat = prepare_catalog(rxnorm_df, RXNORM)

    snomed_cat.to_parquet(os.path.join(args.out_dir, SNOMED_CAT_NAME), index=False)
    rxnorm_cat.to_parquet(os.path.join(args.out_dir, RXNORM_CAT_NAME), index=False)

    sn_vec_path = os.path.join(args.out_dir, "snomed_vectors.f32")
    rx_vec_path = os.path.join(args.out_dir, "rxnorm_vectors.f32")
    sn_vec_mm = np.memmap(sn_vec_path, dtype=np.float32, mode="w+",
                          shape=(len(snomed_cat), dim))
    rx_vec_mm = np.memmap(rx_vec_path, dtype=np.float32, mode="w+",
                          shape=(len(rxnorm_cat), dim))

    # Build indices
    def make_index() -> faiss.Index:
        it = args.index.lower()
        if it == "hnsw":
            return build_hnsw(dim, m=args.hnsw_m, efc=args.hnsw_efc)
        elif it == "ivfpq":
            return build_ivfpq(dim, nlist=args.ivf_nlist, pq_m=args.pq_m, pq_nbits=args.pq_nbits)
        elif it == "flat":
            return build_flat(dim)
        raise ValueError("--index must be: hnsw | ivfpq | flat")

    # --- SNOMED ---
    print(f"[Build] SNOMED rows: {len(snomed_cat):,}")
    sn_index = make_index()
    if args.index.lower() == "ivfpq":
        sample_sz = min(200_000, len(snomed_cat))
        sample_idx = np.linspace(0, len(snomed_cat) - 1, num=sample_sz, dtype=int)
        sample_vecs = embed_texts(model, snomed_cat.loc[sample_idx, "STR"].tolist(), batch_size=args.batch)
        train_ivfpq_if_needed(sn_index, sample_vecs)
    sn_index = faiss.IndexIDMap2(sn_index)
    for start in tqdm(range(0, len(snomed_cat), args.batch), desc="[Build] SNOMED add"):
        end = min(start + args.batch, len(snomed_cat))
        vecs = embed_texts(model, snomed_cat.loc[start:end-1, "STR"].tolist(), batch_size=args.batch)
        ids  = snomed_cat.loc[start:end-1, "row_id"].to_numpy(dtype=np.int64)
        sn_index.add_with_ids(vecs, ids)
        sn_vec_mm[ids] = vecs
    if args.index.lower() == "hnsw":
        faiss.downcast_index(sn_index.index).hnsw.efSearch = args.hnsw_efs
    faiss.write_index(sn_index, os.path.join(args.out_dir, SNOMED_INDEX_NAME))
    print("[Build] Saved SNOMED index.")

    # --- Barrier ---
    try:
        del vecs, ids
    except Exception:
        pass
    clear_cuda_memory("after SNOMED phase")
    model = model_roundtrip_cpu_gpu(model)

    # --- RXNORM ---
    print(f"[Build] RxNorm rows: {len(rxnorm_cat):,}")
    rx_index = make_index()
    if args.index.lower() == "ivfpq":
        sample_sz = min(200_000, len(rxnorm_cat))
        sample_idx = np.linspace(0, len(rxnorm_cat) - 1, num=sample_sz, dtype=int)
        sample_vecs = embed_texts(model, rxnorm_cat.loc[sample_idx, "STR"].tolist(), batch_size=args.batch)
        train_ivfpq_if_needed(rx_index, sample_vecs)
    rx_index = faiss.IndexIDMap2(rx_index)
    for start in tqdm(range(0, len(rxnorm_cat), args.batch), desc="[Build] RxNorm add"):
        end = min(start + args.batch, len(rxnorm_cat))
        vecs = embed_texts(model, rxnorm_cat.loc[start:end-1, "STR"].tolist(), batch_size=args.batch)
        ids  = rxnorm_cat.loc[start:end-1, "row_id"].to_numpy(dtype=np.int64)
        rx_index.add_with_ids(vecs, ids)
        rx_vec_mm[ids] = vecs
    if args.index.lower() == "hnsw":
        faiss.downcast_index(rx_index.index).hnsw.efSearch = args.hnsw_efs
    faiss.write_index(rx_index, os.path.join(args.out_dir, RXNORM_INDEX_NAME))
    print("[Build] Saved RxNorm index.")

    sn_vec_mm.flush()
    rx_vec_mm.flush()

    # --- STY vocabulary embeddings for fast STY similarity ---
    sty_vocab = sorted(list(set(snomed_cat["STY"].dropna().astype(str)) | set(rxnorm_cat["STY"].dropna().astype(str))))
    np.save(os.path.join(args.out_dir, STY_EMB_NPY),
            embed_texts(model, sty_vocab, batch_size=min(512, args.batch), normalize=True))
    with open(os.path.join(args.out_dir, STY_VOCAB_JSON), "w", encoding="utf-8") as f:
        json.dump({"sty_vocab": sty_vocab}, f, indent=2)

    meta = {
        "model": args.model,
        "dim": dim,
        "index_type": args.index.lower(),
        "metric": "ip_cosine_normalized",
        "hnsw": {"M": args.hnsw_m, "efConstruction": args.hnsw_efc, "efSearch": args.hnsw_efs} if args.index.lower()=="hnsw" else None,
        "ivfpq": {"nlist": args.ivf_nlist, "pq_m": args.pq_m, "pq_nbits": args.pq_nbits} if args.index.lower()=="ivfpq" else None,
        "batch": args.batch,
        "files": {
            "snomed_index": SNOMED_INDEX_NAME,
            "rxnorm_index": RXNORM_INDEX_NAME,
            "snomed_catalog": SNOMED_CAT_NAME,
            "rxnorm_catalog": RXNORM_CAT_NAME,
            "sty_vocab_json": STY_VOCAB_JSON,
            "sty_embeddings": STY_EMB_NPY,
            "snomed_vectors": "snomed_vectors.f32",
            "rxnorm_vectors": "rxnorm_vectors.f32",
        }
    }
    with open(os.path.join(args.out_dir, META_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Build] Done. Artifacts in: {args.out_dir}")

# -------------------- LOAD --------------------
def load_bundle(index_dir: str):
    with open(os.path.join(index_dir, META_NAME), "r", encoding="utf-8") as f:
        meta = json.load(f)
    sn_index = faiss.read_index(os.path.join(index_dir, meta["files"]["snomed_index"]))
    rx_index = faiss.read_index(os.path.join(index_dir, meta["files"]["rxnorm_index"]))
    sn_cat = pd.read_parquet(os.path.join(index_dir, meta["files"]["snomed_catalog"]))
    rx_cat = pd.read_parquet(os.path.join(index_dir, meta["files"]["rxnorm_catalog"]))
    with open(os.path.join(index_dir, meta["files"]["sty_vocab_json"]), "r", encoding="utf-8") as f:
        sty_vocab = json.load(f)["sty_vocab"]
    sty_emb = np.load(os.path.join(index_dir, meta["files"]["sty_embeddings"]))
    model = load_model(meta["model"])
    sn_vecs = np.memmap(os.path.join(index_dir, meta["files"]["snomed_vectors"]),
                        dtype=np.float32, mode="r", shape=(len(sn_cat), meta["dim"]))
    rx_vecs = np.memmap(os.path.join(index_dir, meta["files"]["rxnorm_vectors"]),
                        dtype=np.float32, mode="r", shape=(len(rx_cat), meta["dim"]))
    return {
        "meta": meta,
        "model": model,
        "snomed": {"index": sn_index, "catalog": sn_cat, "vecs": sn_vecs},
        "rxnorm": {"index": rx_index, "catalog": rx_cat, "vecs": rx_vecs},
        "sty": {"vocab": sty_vocab, "emb": sty_emb}
    }

# -------------------- Utilities: normalize cos to [0,1] --------------------
def cos_to_01(x: float) -> float:
    return max(0.0, min(1.0, 0.5 * (x + 1.0)))

# -------------------- Scoring building blocks --------------------
def faiss_search(index: faiss.Index, vecs: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(vecs, topk)

def reconstruct_vec(index: faiss.Index, row_id: int, d: int) -> Optional[np.ndarray]:
    try:
        v = np.empty(d, dtype=np.float32)
        index.reconstruct(row_id, v)
        # ensure normalized (FAISS stored were normalized)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v
    except Exception:
        return None

# Build hits: turn FAISS (D,I) to list of rows with metadata & score
def build_row_hits(D_row: np.ndarray, I_row: np.ndarray, catalog: pd.DataFrame) -> List[Dict]:
    hits = []
    for idx, score in zip(I_row, D_row):
        idx = int(idx)
        if idx == -1: 
            continue
        row = catalog.loc[catalog["row_id"] == idx].iloc[0]
        hits.append({
            "row_id": idx,
            "CODE": str(row["CODE"]),
            "System": row["System"],
            "STR": row["STR"],
            "STY": row["STY"],
            "TTY": row["TTY"],
            "score": float(score)
        })
    return hits

# -------------------- ADVANCED PIPELINE (per-query) --------------------
@dataclass
class AdvancedWeights:
    w_desc: float = 0.30
    w_kw:   float = 0.40
    w_direct: float = 0.20
    w_sty:  float = 0.10

@dataclass
class AdvancedParams:
    semantic_top_desc: int = 500
    semantic_top_direct: int = 500
    semantic_top_kw_each: int = 100
    fuzzy_pool: int = 500
    top_pool_after_fuzzy: int = 250
    final_llm_candidates: int = 30
    final_output_k: int = 10
    fuzzy_workers: int = -1   # RapidFuzz cdist workers (-1 = all cores)

def llm_expand_query(llm: LLMClient, text: str, entity_type: str, sty_inventory: List[str]) -> List[Dict]:
    sys_prompt = EXPAND_SYSTEM_PROMPT.format(sty_inventory="\n".join(f"- {s}" for s in sty_inventory))
    user_prompt = EXPAND_USER_PROMPT.format(text=text, entity_type=entity_type)
    out = llm.generate(sys_prompt, user_prompt)
    return extract_xml_candidates(out)

def llm_rerank_codes(llm: LLMClient,
                     text: str,
                     entity_type: str,
                     expanded_summary: str,
                     candidates_blob: str,
                     final_k: int) -> List[str]:
    sys_prompt = RERANK_SYSTEM_PROMPT
    user_prompt = RERANK_USER_PROMPT.format(text=text, entity_type=entity_type,
                                            expanded_summary=expanded_summary,
                                            candidates_blob=candidates_blob,
                                            final_k=final_k)
    out = llm.generate(sys_prompt, user_prompt)
    choices = extract_xml_choices(out)
    return [c["code"] for c in choices][:final_k]

def build_expanded_summary(cands: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(cands, 1):
        ak = ", ".join(c.get("alternate_keywords", [])[:10])
        stys = ", ".join(c.get("possible_stys", [])[:8])
        desc = c.get("description", "")
        lines.append(f"[Candidate {i}] AK: {ak}\nSTY: {stys}\nDESC: {desc}")
    return "\n".join(lines)

def precompute_sty_scores_map(pred_stys: List[str],
                              sty_vocab: List[str],
                              sty_emb: np.ndarray) -> Dict[str, float]:
    """
    Return a dict: candidate STY string -> [0,1] score,
    computed as max cosine(sim(pred_stys), cand_sty) using the prebuilt sty_emb.
    Assumes sty_emb rows are already L2-normalized.
    """
    if not pred_stys:
        return {}

    sty2idx = {s: i for i, s in enumerate(sty_vocab)}
    idxs = [sty2idx[s] for s in pred_stys if s in sty2idx]
    if not idxs:
        return {}

    P = sty_emb[idxs]                 # (m, d)
    # max over predicted STYs against all vocab vectors in one shot
    # result shape: (|vocab|,)
    max_sims = (P @ sty_emb.T).max(axis=0)
    # map to [0,1] and return as dict
    return {sty_vocab[j]: cos_to_01(float(max_sims[j])) for j in range(len(sty_vocab))}

def compute_sty_score(pred_stys: List[str], sty_vocab: List[str], sty_emb: np.ndarray,
                      model: SentenceTransformer, candidate_sty: str) -> float:
    if not pred_stys or not candidate_sty:
        return 0.0
    # map candidate sty embedding
    try:
        idx = sty_vocab.index(candidate_sty)
        cand_vec = sty_emb[idx:idx+1]  # (1,d)
    except ValueError:
        return 0.0
    # embed predicted list and take max similarity
    pred_vecs = embed_texts(model, pred_stys, batch_size=min(64, len(pred_stys)))
    pred_vecs = pred_vecs / np.linalg.norm(pred_vecs, axis=1, keepdims=True).clip(min=1e-9)
    cand_vec = cand_vec / np.linalg.norm(cand_vec, axis=1, keepdims=True).clip(min=1e-9)
    sims = (pred_vecs @ cand_vec.T).flatten()
    return cos_to_01(float(np.max(sims))) if sims.size else 0.0

def fuzzy_score_for_str(s: str, query: str, alt_keywords: List[str]) -> float:
    if not HAVE_RAPIDFUZZ:
        return 0.0
    best = fuzz.token_set_ratio(query, s) / 100.0
    for kw in alt_keywords:
        best = max(best, fuzz.token_set_ratio(kw, s) / 100.0)
    return best

# --- Fast fuzzy scoring (batched + parallel) ---
def _rf_norm(s: str) -> str:
    try:
        from rapidfuzz.utils import default_process
        return default_process(s or "")
    except Exception:
        return re.sub(r"\W+", " ", str(s).lower()).strip()

def fuzzy_scores_max(query: str,
                     alt_keywords: List[str],
                     choices: List[str],
                     kw_limit: int = 8,
                     workers: int = -1,
                     prefilter: int = 200,
                     use_two_stage: bool = True) -> np.ndarray:
    """
    Return an array of max fuzzy scores in [0,1] for each choice string,
    computed against {query} ∪ top-{kw_limit} alt keywords.

    - Batch + parallel via RapidFuzz
    - Optional two-stage: ratio prefilter -> token_set_ratio refine
    """
    if not HAVE_RAPIDFUZZ or not choices:
        return np.zeros(len(choices), dtype=np.float32)

    from rapidfuzz import process as rf_process, fuzz as rf_fuzz

    # Prepare anchors (query + limited KWs), normalized & deduped
    anchors = [_rf_norm(query)]
    for kw in alt_keywords:
        t = _rf_norm(kw)
        if len(t) < 3:            # drop trivially short tokens
            continue
        if t not in anchors:
            anchors.append(t)
        if len(anchors) >= 1 + kw_limit:
            break

    choices_proc = [_rf_norm(c) for c in choices]
    N = len(choices_proc)

    # Stage 1: quick prefilter (ratio) to shrink the set
    if use_two_stage:
        keep = np.zeros(N, dtype=bool)
        lim = min(prefilter, N)
        for a in anchors:
            # Parallel, C++ backend; 'workers' is supported here
            arr = rf_process.cdist([a], choices_proc, scorer=rf_fuzz.ratio, workers=workers)[0]  # 0..100
            # Take top 'lim' indices for this anchor
            if lim < N:
                idxs_top = np.argpartition(arr, -lim)[-lim:]
            else:
                idxs_top = np.arange(N, dtype=int)
            keep[idxs_top] = True

        idxs = np.flatnonzero(keep)
        if idxs.size == 0:
            return np.zeros(N, dtype=np.float32)

        reduced = [choices_proc[i] for i in idxs]
    else:
        idxs = np.arange(N, dtype=int)
        reduced = choices_proc

    # Stage 2: heavy scorer on reduced set (batched)
    best = np.zeros(len(reduced), dtype=np.float32)
    for a in anchors:
        arr = rf_process.cdist([a], reduced, scorer=rf_fuzz.token_set_ratio, workers=workers)[0]
        # no divide yet; keep 0..100 for precision, cast to float32
        np.maximum(best, arr.astype(np.float32), out=best)

    scores = np.zeros(N, dtype=np.float32)
    scores[idxs] = best / 100.0   # normalize to 0..1 like before
    return scores

# -------- Async wrappers for blocking compute --------
_embed_sem = None
_faiss_sem = None

def _init_sems():
    global _embed_sem, _faiss_sem
    # limit concurrent GPU-heavy work; tune to your setup
    max_gpu_jobs = torch.cuda.device_count() or 1
    emb_slots = (2 * max_gpu_jobs) if torch.cuda.is_available() else max(1, multiprocessing.cpu_count() // 2)

    # Honor shared CPU pool size if set (from run_batch runner)
    pool_cap_env = os.environ.get("CPU_POOL", "")
    try:
        pool_cap = int(pool_cap_env) if pool_cap_env else None
    except Exception:
        pool_cap = None
    if pool_cap:
        # keep some headroom so BLAS/FAISS threads can run
        emb_slots = min(emb_slots, max(1, pool_cap // 2))

    faiss_slots = max(2, emb_slots * 2)
    _embed_sem = asyncio.Semaphore(emb_slots)
    _faiss_sem = asyncio.Semaphore(faiss_slots)

async def embed_texts_async(model, texts, batch_size=1024, normalize=True):
    if _embed_sem is None:
        _init_sems()
    async with _embed_sem:
        # offload to a thread so the event loop can run other tasks
        return await asyncio.to_thread(embed_texts, model, texts, batch_size, normalize)

async def faiss_search_async(index: faiss.Index, vecs: np.ndarray, topk: int):
    if _faiss_sem is None:
        _init_sems()
    async with _faiss_sem:
        return await asyncio.to_thread(faiss_search, index, vecs, topk)

async def reconstruct_vec_async(index: faiss.Index, row_id: int, d: int):
    # cheap, but still offload to avoid blocking
    return await asyncio.to_thread(reconstruct_vec, index, row_id, d)

async def fuzzy_scores_max_async(query, alt_keywords, choices, **kwargs):
    return await asyncio.to_thread(fuzzy_scores_max, query, alt_keywords, choices, **kwargs)

def advanced_match_one(query: str,
                       entity_type: str,
                       bundle: Dict,
                       llm: Optional[LLMClient],
                       use_llm_clean: bool,
                       use_llm_rerank: bool,
                       weights: AdvancedWeights,
                       params: AdvancedParams,
                       rerank_mode: str) -> Tuple[List[Dict], str]:
    """
    Returns (final_ranked_rows, reason_string). Each row dict has System/CODE/STR/STY/TTY/score.
    Searches BOTH systems and merges. Optimized to avoid per-row STY embeddings and to batch KW searches.
    """
    model = bundle["model"]
    sn_index, sn_cat = bundle["snomed"]["index"], bundle["snomed"]["catalog"]
    rx_index, rx_cat = bundle["rxnorm"]["index"], bundle["rxnorm"]["catalog"]
    sty_vocab, sty_emb = bundle["sty"]["vocab"], bundle["sty"]["emb"]

    # ---- 1) LLM expansion ----
    if use_llm_clean and llm is not None:
        exp_cands = llm_expand_query(llm, query, entity_type, sty_vocab)
    else:
        exp_cands = [{"alternate_keywords": [], "description": "", "possible_stys": []}]
    if not exp_cands:
        exp_cands = [{"alternate_keywords": [], "description": "", "possible_stys": []}]
    exp_cands = exp_cands[:5]
    expanded_summary = build_expanded_summary(exp_cands)

    # ---- 2) Embeddings for signals ----
    q_vec = embed_texts(model, [query], batch_size=4)[0:1]

    row_scores: Dict[Tuple[str,int], Dict[str, Any]] = {}  # key=(system,row_id)

    def ensure_store(system: str, r: Dict) -> Dict[str, Any]:
        key = (system, r["row_id"])
        if key not in row_scores:
            row_scores[key] = {**r, "score_components": {}, "composite": 0.0}
        return row_scores[key]

    def search_one_system(system: str, vec: np.ndarray, topk: int) -> List[Dict]:
        idx = sn_index if system == SNOMED else rx_index
        cat = sn_cat if system == SNOMED else rx_cat
        D, I = faiss_search(idx, vec, topk)
        return build_row_hits(D[0], I[0], cat)

    def batch_search_one_system(system: str, vecs: np.ndarray, topk: int) -> List[List[Dict]]:
        if vecs.shape[0] == 0:
            return []
        idx = sn_index if system == SNOMED else rx_index
        cat = sn_cat if system == SNOMED else rx_cat
        D, I = faiss_search(idx, vecs, topk)  # vecs: (m,d)
        return [build_row_hits(D[i], I[i], cat) for i in range(len(vecs))]

    # ----- Direct query (do once) -----
    for sys in (SNOMED, RXNORM):
        hits = search_one_system(sys, q_vec, params.semantic_top_direct)
        for r in hits:
            st = ensure_store(sys, r)
            sc = st["score_components"]
            sc["direct"] = max(sc.get("direct", 0.0), cos_to_01(r["score"]))

    # For each candidate: description, alt keywords, STY map
    for cand in exp_cands:
        alt_kws: List[str] = cand.get("alternate_keywords", [])
        desc: str = cand.get("description", "")
        pred_stys: List[str] = cand.get("possible_stys", [])

        # ----- description -----
        if desc.strip():
            d_vec = embed_texts(model, [desc], batch_size=4)[0:1]
            for sys in (SNOMED, RXNORM):
                hits = search_one_system(sys, d_vec, params.semantic_top_desc)
                for r in hits:
                    st = ensure_store(sys, r)
                    sc = st["score_components"]
                    sc["desc"] = max(sc.get("desc", 0.0), cos_to_01(r["score"]))

        # ----- keywords (batched) -----
        kw_vecs = embed_texts(model, alt_kws, batch_size=min(64, len(alt_kws))) if alt_kws else np.zeros((0, q_vec.shape[1]), np.float32)
        for sys in (SNOMED, RXNORM):
            batched = batch_search_one_system(sys, kw_vecs, params.semantic_top_kw_each)
            for hits in batched:
                for r in hits:
                    st = ensure_store(sys, r)
                    sc = st["score_components"]
                    sc["kw"] = max(sc.get("kw", 0.0), cos_to_01(r["score"]))

        # ----- STY similarity (precomputed map; take max across candidates) -----
        if pred_stys:
            sty_map = precompute_sty_scores_map(pred_stys, sty_vocab, sty_emb)
            for st in row_scores.values():
                prev = st["score_components"].get("sty", 0.0)
                st["score_components"]["sty"] = max(prev, sty_map.get(st.get("STY", ""), 0.0))

    # Fill missing components & composite
    for st in row_scores.values():
        sc = st["score_components"]
        sc.setdefault("desc", 0.0)
        sc.setdefault("kw", 0.0)
        sc.setdefault("direct", 0.0)
        sc.setdefault("sty", 0.0)
        st["composite"] = (weights.w_desc * sc["desc"] +
                           weights.w_kw   * sc["kw"] +
                           weights.w_direct * sc["direct"] +
                           weights.w_sty  * sc["sty"])

    # ---- 3) Take top pool and fuzzy re-rank ----
    pooled_rows = sorted(row_scores.values(), key=lambda x: x["composite"], reverse=True)[:params.fuzzy_pool]

    if HAVE_RAPIDFUZZ and pooled_rows:
        # Build a compact keyword bag
        all_kws = []
        for c in exp_cands:
            all_kws.extend(c.get("alternate_keywords", []))
        # Compute batched fuzzy scores (parallel)
        choices = [r["STR"] for r in pooled_rows]
        fuzzy_arr = fuzzy_scores_max(
            query=query,
            alt_keywords=list(dict.fromkeys(all_kws))[:30],   # still cap total collected
            choices=choices,
            kw_limit=8,            # tuneable; 6–10 is a good sweet spot
            workers=-1,            # all CPU cores
            prefilter=max(200, params.top_pool_after_fuzzy * 2),  # size of the reduced set
            use_two_stage=True
        )
        for r, sc in zip(pooled_rows, fuzzy_arr):
            r["fuzzy"] = float(sc)
        pooled_rows.sort(key=lambda x: x["fuzzy"], reverse=True)

    top_rows = pooled_rows[:params.top_pool_after_fuzzy]


    # ---- 4) Aggregate per code ----
    cat_by_sys = {SNOMED: sn_cat, RXNORM: rx_cat}
    idx_by_sys = {SNOMED: sn_index, RXNORM: rx_index}

    have_desc = any(c.get("description", "").strip() for c in exp_cands)
    desc_vec = embed_texts(model, [max(exp_cands, key=lambda c: len(c.get("description", "")))["description"]], 4)[0:1] if have_desc else None
    all_kws = []
    for c in exp_cands: all_kws += c.get("alternate_keywords", [])
    kw_vecs_all = embed_texts(model, list(dict.fromkeys(all_kws)), batch_size=min(64, max(1, len(all_kws)))) if all_kws else None
    pred_stys_all = []
    for c in exp_cands: pred_stys_all += c.get("possible_stys", [])
    pred_stys_all = list(dict.fromkeys(pred_stys_all))
    sty_map_all = precompute_sty_scores_map(pred_stys_all, sty_vocab, sty_emb)

    def composite_for_rowvec(r: Dict, vec: np.ndarray) -> float:
        sc_desc = float((desc_vec @ vec.reshape(1,-1).T).item()) if desc_vec is not None else 0.0
        sc_kw = float(np.max(kw_vecs_all @ vec.reshape(1,-1).T)) if (kw_vecs_all is not None and kw_vecs_all.shape[0] > 0) else 0.0
        sc_dir = float((q_vec @ vec.reshape(1,-1).T).item())
        sc_sty = sty_map_all.get(r.get("STY", ""), 0.0)
        return (weights.w_desc * cos_to_01(sc_desc) +
                weights.w_kw   * cos_to_01(sc_kw)   +
                weights.w_direct * cos_to_01(sc_dir) +
                weights.w_sty  * sc_sty)

    # Bucket rows by (system, code)
    codes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in top_rows:
        key = (r["System"], r["CODE"])
        codes.setdefault(key, {"rows": [], "rows_top": []})
        codes[key]["rows_top"].append(r)

    d = bundle["meta"]["dim"]
    for (system, code), bucket in codes.items():
        cat = cat_by_sys[system]
        idx = idx_by_sys[system]
        all_rows = cat.loc[cat["CODE"] == code]
        bucket["rows"] = []
        for _, row in all_rows.iterrows():
            rid = int(row["row_id"])
            vec = reconstruct_vec(idx, rid, d)
            if vec is None:
                continue
            rdict = {"row_id": rid, "System": system, "CODE": code, "STR": row["STR"], "STY": row["STY"], "TTY": row["TTY"]}
            rdict["composite"] = composite_for_rowvec(rdict, vec)
            bucket["rows"].append(rdict)

    ranked_codes = []
    for (system, code), bucket in codes.items():
        rows_all = bucket["rows"]
        rows_top = bucket["rows_top"]
        if not rows_top or not rows_all:
            continue
        avg_all = float(np.mean([r["composite"] for r in rows_all]))
        avg_500 = float(np.mean([r["composite"] for r in rows_top]))
        denom = max(1, len(rows_all))
        pct = (len(rows_top) / denom) * 100.0
        pct_safe = max(1.0001, pct)
        boost = math.sqrt(math.log10(pct_safe))
        ranked_codes.append({"System": system, "CODE": code,
                             "avg_all": avg_all, "avg_500": avg_500,
                             "pct_in_top500": pct, "final": avg_all * avg_500 * boost})

    ranked_codes.sort(key=lambda x: x["final"], reverse=True)
    topN_codes = ranked_codes[:params.final_llm_candidates]

    def best_name_for_code(system: str, code: str) -> Dict:
        cat = cat_by_sys[system]
        subset = cat.loc[cat["CODE"] == code].copy()
        subset["tty_rank"] = subset["TTY"].map({"PT": 0, "FN": 1, "SCD": 1, "SBD": 1}).fillna(5)
        win = subset.sort_values(["tty_rank"]).iloc[0]
        return {"STR": win["STR"], "STY": win["STY"], "TTY": win["TTY"]}

    topN_rows = [{**c, **best_name_for_code(c["System"], c["CODE"])} for c in topN_codes]

    if use_llm_rerank and llm is not None and len(topN_rows) > 0:
        cand_blob = "\n".join([f"{r['System']} | {r['CODE']} | {r['STY']} | {r['TTY']} | {r['STR']}" for r in topN_rows])
        final_codes = llm_rerank_codes(llm, query, entity_type, expanded_summary, cand_blob, params.final_output_k)
        best_by_code: Dict[str, Tuple[str, str, float]] = {}
        for r in topN_rows:
            tup = best_by_code.get(r["CODE"])
            if (tup is None) or (r["final"] > tup[2]):
                best_by_code[r["CODE"]] = (r["System"], r["CODE"], r.get("final", 0.0))
        final_rows = []
        for code in final_codes:
            if code in best_by_code:
                sys, c, _ = best_by_code[code]
                rep = best_name_for_code(sys, c)
                final_rows.append({"System": sys, "CODE": c, **rep})
        seen = {(r["System"], r["CODE"]) for r in final_rows}
        for r in topN_rows:
            if (r["System"], r["CODE"]) not in seen and len(final_rows) < params.final_output_k:
                final_rows.append({"System": r["System"], "CODE": r["CODE"],
                                   "STR": r["STR"], "STY": r["STY"], "TTY": r["TTY"]})
        return final_rows, "LLM rerank (RxNorm preferred for meds; SNOMED otherwise)"

    final_rows = [{"System": r["System"], "CODE": r["CODE"], "STR": r["STR"], "STY": r["STY"], "TTY": r["TTY"]}
                  for r in topN_rows[:params.final_output_k]]
    return final_rows, "Advanced scoring without LLM rerank"


async def advanced_match_one_async(query: str,
                                   entity_type: str,
                                   bundle: Dict,
                                   llm: Optional[AsyncLLMClient],
                                   use_llm_clean: bool,
                                   use_llm_rerank: bool,
                                   weights: AdvancedWeights,
                                   params: AdvancedParams,
                                   rerank_mode: str) -> Tuple[List[Dict], str]:
    """
    Returns (final_ranked_rows, reason_string). Each row dict has System/CODE/STR/STY/TTY/score.
    This function searches BOTH systems and merges.
    """

    nlc = '\n'
    print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Started matching process for: {query[:60].replace(nlc, ' ')}...")

    model = bundle["model"]
    dim = model.get_sentence_embedding_dimension()
    sn_index, sn_cat = bundle["snomed"]["index"], bundle["snomed"]["catalog"]
    rx_index, rx_cat = bundle["rxnorm"]["index"], bundle["rxnorm"]["catalog"]
    sty_vocab, sty_emb = bundle["sty"]["vocab"], bundle["sty"]["emb"]

    # ---- 1) LLM expansion (multiple candidates allowed) ----
    if use_llm_clean and llm is not None:
        exp_cands = await llm_expand_query_async(llm, query, entity_type, bundle["sty"]["vocab"])
    else:
        exp_cands = [{"alternate_keywords": [], "description": "", "possible_stys": []}]

    # If somehow empty:
    if not exp_cands:
        exp_cands = [{"alternate_keywords": [], "description": "", "possible_stys": []}]

    expanded_summary = build_expanded_summary(exp_cands)

    # ---- 2) Prepare embeddings for signals ----
    # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Preparing embeddings for: {query[:60].replace(nlc, ' ')}...")
    # ---- 2) Prepare embeddings for signals ----
    q_vec = (await embed_texts_async(model, [query], batch_size=4))[0:1]

    # helper: async FAISS search
    async def search_one_system_async(system: str, vec: np.ndarray, topk: int) -> List[Dict]:
        idx = sn_index if system == SNOMED else rx_index
        cat = sn_cat if system == SNOMED else rx_cat
        D, I = await faiss_search_async(idx, vec, topk)
        return build_row_hits(D[0], I[0], cat)

    async def batch_search_one_system_async(system: str, vecs: np.ndarray, topk: int) -> List[List[Dict]]:
        if vecs.shape[0] == 0:
            return []
        idx = sn_index if system == SNOMED else rx_index
        cat = sn_cat if system == SNOMED else rx_cat
        D, I = await faiss_search_async(idx, vecs, topk)
        return [build_row_hits(D[i], I[i], cat) for i in range(len(vecs))]

    def update_scores(system: str, row_list: List[Dict], contrib_name: str, score_getter):
        for r in row_list:
            key = (system, r["row_id"])
            if key not in row_scores:
                row_scores[key] = {**r, "score_components": {}, "composite": 0.0}
            row_scores[key]["score_components"][contrib_name] = score_getter(r)

    # Helper to run a vector search against one system
    def search_one_system(system: str, vec: np.ndarray, topk: int) -> List[Dict]:
        idx = sn_index if system == SNOMED else rx_index
        cat = sn_cat if system == SNOMED else rx_cat
        D,I = faiss_search(idx, vec, topk)
        return build_row_hits(D[0], I[0], cat)
    
    def batch_search_one_system(system: str, vecs: np.ndarray, topk: int) -> List[List[Dict]]:
        if vecs.shape[0] == 0:
            return []
        idx = sn_index if system == SNOMED else rx_index
        cat = sn_cat if system == SNOMED else rx_cat
        D, I = faiss_search(idx, vecs, topk)                      # vecs: (m, d)
        return [build_row_hits(D[i], I[i], cat) for i in range(len(vecs))]

    # For each candidate we will track a big dict of per-row scores and take max across candidates
    row_scores: Dict[Tuple[str,int], Dict[str, Any]] = {}  # key: (system,row_id)

    # For each candidate: description, alt keywords
    for cand in exp_cands:
        alt_kws = cand.get("alternate_keywords", [])
        desc = cand.get("description", "")
        pred_stys = cand.get("possible_stys", [])

        # ----- description -----
        # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Candidate description matching: {desc[:60].replace(nlc, ' ')}...")
        if desc.strip():
            d_vec = (await embed_texts_async(model, [desc], batch_size=4))[0:1]
            for sys in (SNOMED, RXNORM):
                hits = await search_one_system_async(sys, d_vec, params.semantic_top_desc)
                update_scores(sys, hits, "desc", lambda r: cos_to_01(r["score"]))

        kw_vecs = await embed_texts_async(model, alt_kws, batch_size=min(64, len(alt_kws))) \
                if alt_kws else np.zeros((0, q_vec.shape[1]), np.float32)

        for sys in (SNOMED, RXNORM):
            batched_hits = await batch_search_one_system_async(sys, kw_vecs, params.semantic_top_kw_each)
            for hits in batched_hits:
                update_scores(sys, hits, "kw", lambda r: cos_to_01(r["score"]))

        # direct query (do this once per candidate group, as you have)
        for sys in (SNOMED, RXNORM):
            hits = await search_one_system_async(sys, q_vec, params.semantic_top_direct)
            update_scores(sys, hits, "direct", lambda r: cos_to_01(r["score"]))

        # ----- STY similarity -----
        # we don't search by STY; we just compute a STY match score for every row encountered so far
        # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] STY similarity for predicted STYs: {', '.join(pred_stys)[:60].replace(nlc, ' ')}...")
        if pred_stys:
            sty_map = precompute_sty_scores_map(pred_stys, sty_vocab, sty_emb)
            for store in row_scores.values():
                store["score_components"]["sty"] = sty_map.get(store.get("STY",""), 0.0)
        # print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Candidate matching done, found {len(row_scores)} unique rows so far.  ")

    # Fill missing components with zeros
    for store in row_scores.values():
        sc = store["score_components"]
        sc.setdefault("desc", 0.0)
        sc.setdefault("kw", 0.0)
        sc.setdefault("direct", 0.0)
        sc.setdefault("sty", 0.0)
        store["composite"] = (weights.w_desc * sc["desc"] +
                              weights.w_kw   * sc["kw"]   +
                              weights.w_direct * sc["direct"] +
                              weights.w_sty  * sc["sty"])

    # ---- 3) Take top pool and fuzzy re-rank ----
    print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Pooling top {params.fuzzy_pool} rows for fuzzy matching  for: {query[:60].replace(nlc, ' ')}...")
    # ---- 3) Take top pool and fuzzy re-rank ----
    pooled_rows = sorted(row_scores.values(), key=lambda x: x["composite"], reverse=True)[:params.fuzzy_pool]

    if HAVE_RAPIDFUZZ and pooled_rows:
        all_kws = []
        for c in exp_cands:
            all_kws.extend(c.get("alternate_keywords", [])[:30])
        choices = [r["STR"] for r in pooled_rows]
        fuzzy_arr = await fuzzy_scores_max_async(
            query=query,
            alt_keywords=list(dict.fromkeys(all_kws))[:30],
            choices=choices,
            kw_limit=8,
            workers=int(getattr(params, "fuzzy_workers", -1)),
            prefilter=max(200, params.top_pool_after_fuzzy * 2),
            use_two_stage=True
        )
        for r, sc in zip(pooled_rows, fuzzy_arr):
            r["fuzzy"] = float(sc)
        pooled_rows.sort(key=lambda x: x["fuzzy"], reverse=True)

    top_rows = pooled_rows[:params.top_pool_after_fuzzy]

    # ---- 4) Aggregate to per-code final score ----
    # Build helper maps
    print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Aggregating top {len(top_rows)} rows to codes for: {query[:60].replace(nlc, ' ')}...")
    cat_by_sys = {SNOMED: sn_cat, RXNORM: rx_cat}
    idx_by_sys = {SNOMED: sn_index, RXNORM: rx_index}

    # Precompute query representations used to recompute composite for reconstructed rows
    # We reuse the best candidate (max signals) approach: just re-evaluate components that exist (desc/kw/direct/sty)
    have_desc = any(c.get("description","").strip() for c in exp_cands)
    desc_vec = (await embed_texts_async(model, [max(exp_cands, key=lambda c: len(c.get("description","")))["description"]], 4))[0:1] \
            if have_desc else None

    all_kws = []
    for c in exp_cands: all_kws += c.get("alternate_keywords", [])
    kw_vecs = await embed_texts_async(model, list(dict.fromkeys(all_kws)),
                                    batch_size=min(64, max(1,len(all_kws)))) if all_kws else None
    
    pred_stys_all = []
    for c in exp_cands: pred_stys_all += c.get("possible_stys", [])
    pred_stys_all = list(dict.fromkeys(pred_stys_all))

    sty_map_all = precompute_sty_scores_map(pred_stys_all, sty_vocab, sty_emb)

    vec_by_sys = {SNOMED: bundle["snomed"]["vecs"], RXNORM: bundle["rxnorm"]["vecs"]}

    def _cos01(arr: np.ndarray) -> np.ndarray:
        # vectorized cos_to_01 for arrays
        return np.clip(0.5 * (arr + 1.0), 0.0, 1.0)

    # Gather all candidate codes present in top_rows
    codes = {}
    for r in top_rows:
        code = (r["System"], r["CODE"])
        codes.setdefault(code, {"rows": [], "rows_top": []})
        codes[code]["rows_top"].append(r)

    # Batched composite per code using memmapped vectors
    for (system, code), bucket in codes.items():
        cat = cat_by_sys[system]
        vecs_mm = vec_by_sys[system]
        all_rows = cat.loc[cat["CODE"] == code].copy()
        bucket["rows"] = []

        if all_rows.empty:
            continue

        row_ids = all_rows["row_id"].to_numpy(dtype=np.int64)
        row_vecs = vecs_mm[row_ids]                         # (num_rows, d)

        # components (all cosine on normalized embeddings)
        sc_dir = (row_vecs @ q_vec.T).ravel() if q_vec is not None else 0.0
        sc_desc = (row_vecs @ desc_vec.T).ravel() if desc_vec is not None else 0.0
        sc_kw = (row_vecs @ kw_vecs.T).max(axis=1) if (kw_vecs is not None and kw_vecs.shape[0] > 0) else 0.0

        sc_dir01  = _cos01(sc_dir)
        sc_desc01 = _cos01(sc_desc) if isinstance(sc_desc, np.ndarray) else 0.0
        sc_kw01   = _cos01(sc_kw)   if isinstance(sc_kw,   np.ndarray) else 0.0

        sc_sty = np.fromiter(
            (sty_map_all.get(str(sty), 0.0) for sty in all_rows["STY"]),
            count=len(all_rows), dtype=np.float32
        )

        comp = (weights.w_desc * sc_desc01 +
                weights.w_kw   * sc_kw01   +
                weights.w_direct * sc_dir01 +
                weights.w_sty  * sc_sty).astype(np.float32)

        bucket["rows"] = [{
            "row_id": int(rid),
            "System": system,
            "CODE": code,
            "STR": str_str,
            "STY": str_sty,
            "TTY": str_tty,
            "composite": float(c)
        } for rid, str_str, str_sty, str_tty, c in zip(
            row_ids,
            all_rows["STR"].tolist(),
            all_rows["STY"].tolist(),
            all_rows["TTY"].tolist(),
            comp
        )]

    # Compute final per-code score
    ranked_codes = []
    for (system, code), bucket in codes.items():
        rows_all = bucket["rows"]
        rows_top = bucket["rows_top"]
        if not rows_top or not rows_all:
            continue
        avg_all = float(np.mean([r["composite"] for r in rows_all]))
        avg_500 = float(np.mean([r["composite"] for r in rows_top]))
        # percentage of instances in top 500
        denom = max(1, len(rows_all))
        pct = (len(rows_top) / denom) * 100.0
        pct_safe = max(1.0001, pct)
        boost = math.sqrt(math.log10(pct_safe))
        final_score = avg_all * avg_500 * boost
        ranked_codes.append({
            "System": system, "CODE": code,
            "avg_all": avg_all, "avg_500": avg_500,
            "pct_in_top500": pct, "final": final_score
        })

    ranked_codes.sort(key=lambda x: x["final"], reverse=True)
    topN_codes = ranked_codes[:params.final_llm_candidates]

    # Materialize one representative row (best STR) per code for reporting & LLM rerank
    def best_name_for_code(system: str, code: str) -> Dict:
        cat = cat_by_sys[system]
        subset = cat.loc[cat["CODE"] == code]
        # choose the PT over SY where possible
        subset = subset.copy()
        subset["tty_rank"] = subset["TTY"].map({"PT":0, "FN":1, "SCD":1, "SBD":1}).fillna(5)
        win = subset.sort_values(["tty_rank"]).iloc[0]
        return {"STR": win["STR"], "STY": win["STY"], "TTY": win["TTY"]}

    topN_rows = []
    for c in topN_codes:
        meta_row = best_name_for_code(c["System"], c["CODE"])
        topN_rows.append({**c, **meta_row})

    # Optional LLM final rerank (codes only)
    if use_llm_rerank and llm is not None and len(topN_rows) > 0:
        print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Final LLM reranking among {len(topN_rows)} candidates for: {query[:60].replace(nlc, ' ')}...")
        cand_blob = "\n".join([f"{r['System']} | {r['CODE']} | {r['STY']} | {r['TTY']} | {r['STR']}" for r in topN_rows])
        final_codes = await llm_rerank_codes_async(llm, query, entity_type, expanded_summary, cand_blob, params.final_output_k)
        # Map LLM output order back to code rows (keep those that exist in our topN)
        code_set = {(r["System"], r["CODE"]) for r in topN_rows}
        # LLM returned only code string; pick the (system,code) with best 'final' if multiple systems share code text
        # Build index by code -> best pair
        best_by_code: Dict[str, Tuple[str,str,float]] = {}
        for r in topN_rows:
            tup = best_by_code.get(r["CODE"])
            if (tup is None) or (r["final"] > tup[2]):
                best_by_code[r["CODE"]] = (r["System"], r["CODE"], r["final"])
        final_rows = []
        for code in final_codes:
            if code in best_by_code:
                sys, c, _ = best_by_code[code]
                # retrieve representative row again
                rep = best_name_for_code(sys, c)
                final_rows.append({"System": sys, "CODE": c, **rep})
        # If LLM returns fewer than requested, fill from topN by 'final'
        seen = {(r["System"], r["CODE"]) for r in final_rows}
        for r in topN_rows:
            if (r["System"], r["CODE"]) not in seen and len(final_rows) < params.final_output_k:
                final_rows.append({"System": r["System"], "CODE": r["CODE"], "STR": r["STR"], "STY": r["STY"], "TTY": r["TTY"]})
        reason = "LLM rerank (RxNorm preferred for meds; SNOMED otherwise)"
        return final_rows, reason

    # No LLM rerank: return topN by final score (truncate to topk)
    final_rows = []
    for r in topN_rows[:params.final_output_k]:
        final_rows.append({"System": r["System"], "CODE": r["CODE"], "STR": r["STR"], "STY": r["STY"], "TTY": r["TTY"]})
    reason = "Advanced scoring without LLM rerank"

    print(f"[LLM {datetime.now().strftime(DATE_FORMAT_STR)}] Matching process complete for: {query[:60].replace(nlc, ' ')}...")
    return final_rows, reason

# -------------------- Legacy (non-LLM) selection for single/batch --------------------
def build_hits_simple(query: str, D_row: np.ndarray, I_row: np.ndarray, catalog: pd.DataFrame,
                      rerank: bool, topk: int) -> List[Dict]:
    hits = build_row_hits(D_row, I_row, catalog)
    if rerank and HAVE_RAPIDFUZZ and len(hits) > 1:
        hits.sort(key=lambda r: fuzz.token_set_ratio(query, r["STR"]), reverse=True)
    return hits[:topk]

def choose_system_hits(preferred_system: str,
                       hits_snomed: List[Dict],
                       hits_rxnorm: List[Dict],
                       min_score: float,
                       alt_margin: float) -> Tuple[str, List[Dict], str]:
    top_sn = hits_snomed[0]["score"] if hits_snomed else -1.0
    top_rx = hits_rxnorm[0]["score"] if hits_rxnorm else -1.0
    if preferred_system == SNOMED:
        pref_hits, alt_hits = hits_snomed, hits_rxnorm
        pref_name, alt_name = SNOMED, RXNORM
        pref_top, alt_top = top_sn, top_rx
    else:
        pref_hits, alt_hits = hits_rxnorm, hits_snomed
        pref_name, alt_name = RXNORM, SNOMED
        pref_top, alt_top = top_rx, top_sn
    if pref_top < min_score:
        if alt_top >= min_score:
            return alt_name, alt_hits, "fallback to alt"
        else:
            return pref_name, [], f"no match ≥ {min_score:.2f}"
    if alt_top >= min_score:
        rel_gain = (alt_top - pref_top) / max(1e-9, abs(pref_top))
        if rel_gain >= alt_margin:
            return alt_name, alt_hits, f"switch (gain {rel_gain:.2%})"
    return pref_name, pref_hits, "prefer preferred"

# -------------------- QUERY --------------------
def run_query(args):
    maybe_hf_login(args.hf_token)
    configure_cpu_threading(args.blas_threads, args.faiss_threads)
    blas_ctx = (threadpool_limits(limits=args.blas_threads)
                if (threadpool_limits and args.blas_threads) else nullcontext())
    bundle = load_bundle(args.index_dir)

    if bundle["meta"]["index_type"] == "hnsw":
        for key in ("snomed", "rxnorm"):
            faiss.downcast_index(bundle[key]["index"].index).hnsw.efSearch = args.hnsw_efs

    with blas_ctx:
        llm = None
        if args.use_llm_clean or args.use_llm_rerank:
            llm = LLMClient(LLMConfig(backend=args.llm_backend,
                                    hf_model_id=args.llm_hf_model_id,
                                    gguf_path=args.llm_gguf_path,
                                    max_new_tokens=args.llm_max_new_tokens,
                                    temperature=args.llm_temperature,
                                    top_p=args.llm_top_p))

        if args.use_llm_clean or args.use_llm_rerank:
            # Advanced pipeline on both systems
            rows, reason = advanced_match_one(
                query=args.text,
                entity_type=args.entity_type,
                bundle=bundle,
                llm=llm,
                use_llm_clean=args.use_llm_clean,
                use_llm_rerank=args.use_llm_rerank,
                weights=AdvancedWeights(args.weights_desc, args.weights_kw, args.weights_direct, args.weights_sty),
                params=AdvancedParams(final_output_k=args.topk, fuzzy_workers=args.fuzzy_workers),
                rerank_mode=args.rerank
            )
            print(f"\n[Query][Advanced] chosen ({reason})")
            if not rows:
                print("No match found.")
                return
            best = rows[0]
            print(f"Top-1: CODE={best['CODE']} | DESC={best['STR']} | STY={best['STY']} | TTY={best['TTY']} | SYS={best['System']}")
            print("\nTop-k candidates:")
            for i, r in enumerate(rows, 1):
                print(f"{i:>2}. {r['System']} | {r['CODE']} | {r['STY']} | {r['TTY']} | {r['STR']}")
            return

    # Legacy path (no LLM): your earlier policy
    model = bundle["model"]
    qv = embed_texts(model, [args.text], batch_size=1)
    D_sn, I_sn = bundle["snomed"]["index"].search(qv, args.topk)
    D_rx, I_rx = bundle["rxnorm"]["index"].search(qv, args.topk)

    hits_sn = build_hits_simple(args.text, D_sn[0], I_sn[0], bundle["snomed"]["catalog"], rerank=(args.rerank=="rapidfuzz"), topk=args.topk)
    hits_rx = build_hits_simple(args.text, D_rx[0], I_rx[0], bundle["rxnorm"]["catalog"], rerank=(args.rerank=="rapidfuzz"), topk=args.topk)

    preferred = choose_system(args.entity_type)
    chosen_sys, chosen_hits, reason = choose_system_hits(preferred, hits_sn, hits_rx, args.min_score, args.alt_margin)

    print(f"\n[Query][Legacy] preferred={preferred} -> chosen={chosen_sys} ({reason})")
    if not chosen_hits:
        print(f"No match ≥ {args.min_score:.2f}")
        return
    best = chosen_hits[0]
    print(f"Top-1: CODE={best['CODE']} | DESC={best['STR']} | STY={best['STY']} | TTY={best['TTY']} | SYS={best['System']}")
    print("\nTop-k candidates:")
    for i, r in enumerate(chosen_hits, 1):
        print(f"{i:>2}. {r['System']} | {r['CODE']} | {r['STY']} | {r['TTY']} | {r['STR']}")

# -------------------- BATCH --------------------
def read_input_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError("Unsupported input format. Use .xlsx/.xls or .csv/.tsv")
    cols_norm = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols_norm)
    if REQ_INPUT_COL not in df.columns or REQ_TYPE_COL not in df.columns:
        raise ValueError(f"Input must have columns: '{REQ_INPUT_COL}' and '{REQ_TYPE_COL}'")
    return df

def write_output_table(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path.lower())[1]
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        df.to_csv(path, index=False, sep=sep)
    else:
        raise ValueError("Unsupported output format. Use .xlsx/.xls or .csv/.tsv")

# def run_batch(args):
#     maybe_hf_login(args.hf_token)
#     bundle = load_bundle(args.index_dir)

#     if bundle["meta"]["index_type"] == "hnsw":
#         for key in ("snomed", "rxnorm"):
#             faiss.downcast_index(bundle[key]["index"].index).hnsw.efSearch = args.hnsw_efs

#     df = read_input_table(args.in_file)

#     llm = None
#     if args.use_llm_clean or args.use_llm_rerank:
#         llm = LLMClient(LLMConfig(backend=args.llm_backend,
#                                   hf_model_id=args.llm_hf_model_id,
#                                   gguf_path=args.llm_gguf_path,
#                                   max_new_tokens=args.llm_max_new_tokens,
#                                   temperature=args.llm_temperature,
#                                   top_p=args.llm_top_p))

#     # Prepare columns
#     for col in (OUT_SYS_COL, OUT_CODE_COL, OUT_DESC_COL, OUT_STY_COL, OUT_TTY_COL):
#         if col not in df.columns:
#             df[col] = ""

#     if args.include_topk:
#         for k in range(1, args.topk + 1):
#             for base, col in [
#                 ("System",    f"Output Coding System {k}"),
#                 ("CODE",      f"Output Target Code {k}"),
#                 ("STR",       f"Output Target Description {k}"),
#                 ("STY",       f"Output Semantic Type {k}"),
#                 ("TTY",       f"Output Term Type {k}"),
#             ]:
#                 if col not in df.columns:
#                     df[col] = ""

#     # Process each row
#     for i in tqdm(range(len(df)), desc="[Batch] LLM+ANN"):
#         text = str(df.at[i, REQ_INPUT_COL])
#         etype = str(df.at[i, REQ_TYPE_COL])

#         if args.use_llm_clean or args.use_llm_rerank:
#             rows, _reason = advanced_match_one(
#                 query=text,
#                 entity_type=etype,
#                 bundle=bundle,
#                 llm=llm,
#                 use_llm_clean=args.use_llm_clean,
#                 use_llm_rerank=args.use_llm_rerank,
#                 weights=AdvancedWeights(args.weights_desc, args.weights_kw, args.weights_direct, args.weights_sty),
#                 params=AdvancedParams(final_output_k=args.topk),
#                 rerank_mode=args.rerank
#             )
#             if rows:
#                 top1 = rows[0]
#                 df.at[i, OUT_SYS_COL]  = top1["System"]
#                 df.at[i, OUT_CODE_COL] = top1["CODE"]
#                 df.at[i, OUT_DESC_COL] = top1["STR"]
#                 df.at[i, OUT_STY_COL]  = top1["STY"]
#                 df.at[i, OUT_TTY_COL]  = top1["TTY"]
#                 if args.include_topk:
#                     for k, r in enumerate(rows, 1):
#                         if k > args.topk: break
#                         df.at[i, f"Output Coding System {k}"]      = r["System"]
#                         df.at[i, f"Output Target Code {k}"]        = r["CODE"]
#                         df.at[i, f"Output Target Description {k}"] = r["STR"]
#                         df.at[i, f"Output Semantic Type {k}"]      = r["STY"]
#                         df.at[i, f"Output Term Type {k}"]          = r["TTY"]
#             continue

#         # Legacy path (no LLM)
#         model = bundle["model"]
#         qv = embed_texts(model, [text], batch_size=1)
#         D_sn, I_sn = bundle["snomed"]["index"].search(qv, args.topk)
#         D_rx, I_rx = bundle["rxnorm"]["index"].search(qv, args.topk)
#         hits_sn = build_hits_simple(text, D_sn[0], I_sn[0], bundle["snomed"]["catalog"], rerank=(args.rerank=="rapidfuzz"), topk=args.topk)
#         hits_rx = build_hits_simple(text, D_rx[0], I_rx[0], bundle["rxnorm"]["catalog"], rerank=(args.rerank=="rapidfuzz"), topk=args.topk)
#         preferred = choose_system(etype)
#         chosen_sys, chosen_hits, _reason = choose_system_hits(preferred, hits_sn, hits_rx, args.min_score, args.alt_margin)
#         if chosen_hits:
#             top1 = chosen_hits[0]
#             df.at[i, OUT_SYS_COL]  = top1["System"]
#             df.at[i, OUT_CODE_COL] = top1["CODE"]
#             df.at[i, OUT_DESC_COL] = top1["STR"]
#             df.at[i, OUT_STY_COL]  = top1["STY"]
#             df.at[i, OUT_TTY_COL]  = top1["TTY"]
#             if args.include_topk:
#                 for k, r in enumerate(chosen_hits, 1):
#                     df.at[i, f"Output Coding System {k}"]      = r["System"]
#                     df.at[i, f"Output Target Code {k}"]        = r["CODE"]
#                     df.at[i, f"Output Target Description {k}"] = r["STR"]
#                     df.at[i, f"Output Semantic Type {k}"]      = r["STY"]
#                     df.at[i, f"Output Term Type {k}"]          = r["TTY"]

#     write_output_table(df, args.out_file)
#     print(f"[Batch] Wrote predictions to: {args.out_file}")

def run_batch(args):
    maybe_hf_login(args.hf_token)
    bundle = load_bundle(args.index_dir)

    if bundle["meta"]["index_type"] == "hnsw":
        for key in ("snomed", "rxnorm"):
            faiss.downcast_index(bundle[key]["index"].index).hnsw.efSearch = args.hnsw_efs
    
    configure_cpu_threading(args.blas_threads, args.faiss_threads)
    blas_ctx = (threadpool_limits(limits=args.blas_threads)
                if (threadpool_limits and args.blas_threads) else nullcontext())

    df = read_input_table(args.in_file)

    # Prepare output columns (same as your version) ...
    for col in (OUT_SYS_COL, OUT_CODE_COL, OUT_DESC_COL, OUT_STY_COL, OUT_TTY_COL):
        if col not in df.columns:
            df[col] = ""
    if args.include_topk:
        for k in range(1, args.topk + 1):
            for name in (f"Output Coding System {k}",
                         f"Output Target Code {k}",
                         f"Output Target Description {k}",
                         f"Output Semantic Type {k}",
                         f"Output Term Type {k}"):
                if name not in df.columns:
                    df[name] = ""

    # Build LLM (async)
    llm = None
    if args.use_llm_clean or args.use_llm_rerank:
        backend, hf_id, gguf = pick_llm_backend_and_model(args.llm_backend,
                                                          args.llm_hf_model_id,
                                                          args.llm_gguf_path)
        cfg = LLMConfig(
            backend=backend,
            hf_model_id=hf_id,
            gguf_path=gguf,
            max_new_tokens=args.llm_max_new_tokens,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            vllm_quantization=args.vllm_quantization,
            tp=args.llm_tp,
            n_threads=args.llm_n_threads,
            concurrency=args.llm_concurrency or default_concurrency(),
        )
        llm = AsyncLLMClient(cfg)

    async def process_row(i: int):
        text = str(df.at[i, REQ_INPUT_COL])
        etype = str(df.at[i, REQ_TYPE_COL])
        if args.use_llm_clean or args.use_llm_rerank:
            rows, _ = await advanced_match_one_async(
                query=text,
                entity_type=etype,
                bundle=bundle,
                llm=llm,
                use_llm_clean=args.use_llm_clean,
                use_llm_rerank=args.use_llm_rerank,
                weights=AdvancedWeights(args.weights_desc, args.weights_kw, args.weights_direct, args.weights_sty),
                params=AdvancedParams(final_output_k=args.topk, fuzzy_workers=args.fuzzy_workers),
                rerank_mode=args.rerank
            )
            if rows:
                top1 = rows[0]
                df.at[i, OUT_SYS_COL]  = top1["System"]
                df.at[i, OUT_CODE_COL] = top1["CODE"]
                df.at[i, OUT_DESC_COL] = top1["STR"]
                df.at[i, OUT_STY_COL]  = top1["STY"]
                df.at[i, OUT_TTY_COL]  = top1["TTY"]
                if args.include_topk:
                    for k, r in enumerate(rows, 1):
                        if k > args.topk: break
                        df.at[i, f"Output Coding System {k}"]      = r["System"]
                        df.at[i, f"Output Target Code {k}"]        = r["CODE"]
                        df.at[i, f"Output Target Description {k}"] = r["STR"]
                        df.at[i, f"Output Semantic Type {k}"]      = r["STY"]
                        df.at[i, f"Output Term Type {k}"]          = r["TTY"]
            return

        # legacy (no LLM) — keep your synchronous block
        model = bundle["model"]
        qv = embed_texts(model, [text], batch_size=1)
        D_sn, I_sn = bundle["snomed"]["index"].search(qv, args.topk)
        D_rx, I_rx = bundle["rxnorm"]["index"].search(qv, args.topk)
        hits_sn = build_hits_simple(text, D_sn[0], I_sn[0], bundle["snomed"]["catalog"], rerank=(args.rerank=="rapidfuzz"), topk=args.topk)
        hits_rx = build_hits_simple(text, D_rx[0], I_rx[0], bundle["rxnorm"]["catalog"], rerank=(args.rerank=="rapidfuzz"), topk=args.topk)
        preferred = choose_system(etype)
        chosen_sys, chosen_hits, _ = choose_system_hits(preferred, hits_sn, hits_rx, args.min_score, args.alt_margin)
        if chosen_hits:
            top1 = chosen_hits[0]
            df.at[i, OUT_SYS_COL]  = top1["System"]
            df.at[i, OUT_CODE_COL] = top1["CODE"]
            df.at[i, OUT_DESC_COL] = top1["STR"]
            df.at[i, OUT_STY_COL]  = top1["STY"]
            df.at[i, OUT_TTY_COL]  = top1["TTY"]
            if args.include_topk:
                for k, r in enumerate(chosen_hits, 1):
                    df.at[i, f"Output Coding System {k}"]      = r["System"]
                    df.at[i, f"Output Target Code {k}"]        = r["CODE"]
                    df.at[i, f"Output Target Description {k}"] = r["STR"]
                    df.at[i, f"Output Semantic Type {k}"]      = r["STY"]
                    df.at[i, f"Output Term Type {k}"]          = r["TTY"]

    async def runner():
        loop = asyncio.get_running_loop()
        cpu_pool_size = int(args.cpu_pool or max(4, multiprocessing.cpu_count()))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_pool_size)
        loop.set_default_executor(pool)
        os.environ["CPU_POOL"] = str(cpu_pool_size)  # used by _init_sems()

        rows_conc = int(getattr(args, "rows_concurrency", None)
                or getattr(args, "llm_concurrency", None)
                or (llm.cfg.concurrency if llm else default_concurrency()))

        q = asyncio.Queue()
        for i in range(len(df)):
            await q.put(i)
        for _ in range(rows_conc):
            await q.put(None)  # sentinels

        pbar = tqdm(total=len(df), desc="[Batch] LLM+ANN (async)")

        async def worker():
            while True:
                i = await q.get()
                print("[Worker] Started worker with index:", i)
                if i is None:
                    break
                try:
                    await process_row(i)
                finally:
                    pbar.update(1)

        workers = [asyncio.create_task(worker()) for _ in range(rows_conc)]
        await asyncio.gather(*workers)
        pbar.close()

        pool.shutdown(wait=True)

    with blas_ctx:
        if args.use_llm_clean or args.use_llm_rerank:
            asyncio.run(runner())
        else:
            # legacy loop
            for i in tqdm(range(len(df)), desc="[Batch] Legacy ANN"):
                asyncio.run(process_row(i))  # safe since process_row handles legacy

    write_output_table(df, args.out_file)
    print(f"[Batch] Wrote predictions to: {args.out_file}")

# -------------------- CLI --------------------
def main():
    p = argparse.ArgumentParser(description="Clinical Harmonization Matcher (FAISS + LLM Assist)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # BUILD
    pb = sub.add_parser("build", help="Build FAISS indices & STY embeddings.")
    pb.add_argument("--snomed_parquet", required=True)
    pb.add_argument("--rxnorm_parquet", required=True)
    pb.add_argument("--out_dir", required=True)
    pb.add_argument("--model", default="google/embeddinggemma-300m")
    pb.add_argument("--index", choices=["hnsw", "ivfpq", "flat"], default="hnsw")
    pb.add_argument("--batch", type=int, default=1024)
    pb.add_argument("--hnsw_m", type=int, default=32)
    pb.add_argument("--hnsw_efc", type=int, default=200)
    pb.add_argument("--hnsw_efs", type=int, default=128)
    pb.add_argument("--ivf_nlist", type=int, default=16384)
    pb.add_argument("--pq_m", type=int, default=64)
    pb.add_argument("--pq_nbits", type=int, default=8)
    pb.add_argument("--hf_token", default=None)

    # QUERY
    pq = sub.add_parser("query", help="Single query lookup.")
    pq.add_argument("--index_dir", required=True)
    pq.add_argument("--text", required=True)
    pq.add_argument("--entity_type", required=True)
    pq.add_argument("--topk", type=int, default=10)
    pq.add_argument("--hnsw_efs", type=int, default=128)
    pq.add_argument("--rerank", choices=["none", "rapidfuzz"], default="rapidfuzz")
    pq.add_argument("--min_score", type=float, default=0.30)
    pq.add_argument("--alt_margin", type=float, default=0.15)
    # LLM options
    pq.add_argument("--use_llm_clean", type=lambda s: s.lower()!="false", default=True)
    pq.add_argument("--use_llm_rerank", type=lambda s: s.lower()!="false", default=True)
    pq.add_argument("--llm_backend", choices=["auto","vllm","llama-cpp"], default="auto")
    pq.add_argument("--llm_hf_model_id", default=UNSLOTH_VLLM_OSS_20B)
    pq.add_argument("--llm_gguf_path", default=None)
    pq.add_argument("--llm_max_new_tokens", type=int, default=512)
    pq.add_argument("--llm_temperature", type=float, default=0.1)
    pq.add_argument("--llm_top_p", type=float, default=0.9)
    pq.add_argument("--weights_desc", type=float, default=0.30)
    pq.add_argument("--weights_kw", type=float, default=0.40)
    pq.add_argument("--weights_direct", type=float, default=0.20)
    pq.add_argument("--weights_sty", type=float, default=0.10)
    pq.add_argument("--hf_token", default=None)

    # BATCH
    pba = sub.add_parser("batch", help="Batch process queries from CSV/XLSX.")
    pba.add_argument("--index_dir", required=True)
    pba.add_argument("--in_file", required=True)
    pba.add_argument("--out_file", required=True)
    pba.add_argument("--topk", type=int, default=10)
    pba.add_argument("--include_topk", action="store_true")
    pba.add_argument("--hnsw_efs", type=int, default=128)
    pba.add_argument("--rerank", choices=["none", "rapidfuzz"], default="rapidfuzz")
    pba.add_argument("--min_score", type=float, default=0.30)
    pba.add_argument("--alt_margin", type=float, default=0.15)
    # LLM options
    pba.add_argument("--use_llm_clean", type=lambda s: s.lower()!="false", default=True)
    pba.add_argument("--use_llm_rerank", type=lambda s: s.lower()!="false", default=True)
    pba.add_argument("--llm_backend", choices=["auto","vllm","llama-cpp"], default="auto")
    pba.add_argument("--llm_hf_model_id", default=UNSLOTH_VLLM_OSS_20B)
    pba.add_argument("--llm_gguf_path", default=None)
    pba.add_argument("--llm_max_new_tokens", type=int, default=512)
    pba.add_argument("--llm_temperature", type=float, default=0.1)
    pba.add_argument("--llm_top_p", type=float, default=0.9)
    pba.add_argument("--weights_desc", type=float, default=0.30)
    pba.add_argument("--weights_kw", type=float, default=0.40)
    pba.add_argument("--weights_direct", type=float, default=0.20)
    pba.add_argument("--weights_sty", type=float, default=0.10)
    pba.add_argument("--hf_token", default=None)

    # common LLM scaling knobs
    pq.add_argument("--llm_concurrency", type=int, default=None,
                    help="Max concurrent LLM requests. Default: 32*num_gpus or 2*(cores-1).")
    pq.add_argument("--llm_tp", default="auto",
                    help="Tensor parallel size for vLLM (int or 'auto').")
    pq.add_argument("--llm_n_threads", default="auto",
                    help="Threads per llama-cpp context (int or 'auto').")
    pq.add_argument("--vllm_quantization", default=DEFAULT_VLLM_QUANT,
                    help="Set vLLM quantization (e.g., mxfp4, awq, gptq).")

    pba.add_argument("--llm_concurrency", type=int, default=None)
    pba.add_argument("--llm_tp", default="auto")
    pba.add_argument("--llm_n_threads", default="auto")
    pba.add_argument("--vllm_quantization", default=DEFAULT_VLLM_QUANT)
    pba.add_argument("--rows_concurrency", type=int, default=20,
                 help="Max concurrent rows (pipeline-level). Defaults to llm_concurrency.")

    # >>> UPDATE START
    # CPU / threading knobs
    pq.add_argument("--blas_threads", type=int, default=None,
                    help="Threads for NumPy/BLAS (OpenBLAS/MKL/BLIS).")
    pq.add_argument("--faiss_threads", type=int, default=None,
                    help="OpenMP threads for FAISS.")
    pq.add_argument("--cpu_pool", type=int, default=None,
                    help="Default asyncio thread-pool size for CPU offloaded work (to_thread).")
    pq.add_argument("--fuzzy_workers", type=int, default=-1,
                    help="Workers for RapidFuzz (cdist). -1 = all cores.")

    pba.add_argument("--blas_threads", type=int, default=None)
    pba.add_argument("--faiss_threads", type=int, default=None)
    pba.add_argument("--cpu_pool", type=int, default=None)
    pba.add_argument("--fuzzy_workers", type=int, default=-1)
    # <<< UPDATE END

    args = p.parse_args()

    if args.cmd == "build":
        build_indices(args)
    elif args.cmd == "query":
        run_query(args)
    elif args.cmd == "batch":
        run_batch(args)

if __name__ == "__main__":
    main()