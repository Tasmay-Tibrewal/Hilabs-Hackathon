# Clinical Concept Harmonizer (Semantic + Fuzzy + LLM-Assist)

> HiLabs Hackathon 2025 — end-to-end engine to normalize messy clinical inputs into **RxNorm** (medications) and **SNOMED CT** (diagnoses/procedures/labs) with a hybrid of embedding search, fast fuzzy matching, and optional LLM assist.

---

## Table of contents

* [What this does](#what-this-does)
* [How it works (architecture)](#how-it-works-architecture)
* [Why it’s novel](#why-its-novel)
* [Repo layout](#repo-layout)
* [Quickstart](#quickstart)
* [Build the indices](#build-the-indices)
* [Run: batch or single query](#run-batch-or-single-query)
* [Output format](#output-format)
* [Tuning & performance knobs](#tuning--performance-knobs)
* [Methodology details](#methodology-details)
* [Rules compliance](#rules-compliance)
* [Deliverables checklist (for evaluators)](#deliverables-checklist-for-evaluators)
* [Troubleshooting](#troubleshooting)
* [Notes on data & licensing](#notes-on-data--licensing)

---

## What this does

Given raw, messy clinical inputs (e.g., “paracetamol 500”, “chest xr”, “hcv rna”), the system:

1. Chooses the appropriate target coding system: **RxNorm** for medications; **SNOMED CT** elsewhere.
2. Uses **semantic search** (SentenceTransformers + FAISS) over pre-built indices.
3. Adds **fuzzy matching** (RapidFuzz) for robustness to spelling/abbreviation noise.
4. Optionally uses an **LLM in two places**:

   * **Query expansion**: propose alternate keywords, short description and likely semantic types (STYs).
   * **Final re-ranking**: reorder top candidates with a short justification.
5. Returns **top-k** normalized codes with system, code, description, STY, and TTY — written back to Excel/CSV.

You can run **LLM-off** (fast, fully open-source) or **LLM-on** (more recall on ambiguous inputs). Works on **CPU** via `llama-cpp-python` and **GPU** via `vLLM` (Qwen3-4B by default; auto-selects based on VRAM).

---

## How it works (architecture)

```
Raw input row ─► [Optional LLM: query expansion]
                    • alternate keywords
                    • short description
                    • candidate STYs (from known STY inventory)
                 ┌──────────────────────────────────────────────────────────┐
                 │ Semantic search (FAISS)                                  │
                 │  • direct query vec                                      │
                 │  • description vec                                       │
                 │  • each alternate keyword vec                            │
                 │  • compute STY match via pre-embedded STY vocabulary     │
                 └──────────────────────────────────────────────────────────┘
                                   │
                     Weighted multi-signal score per row
                     (desc/keywords/direct/STY)  →  pool top N rows
                                   │
                 Fast fuzzy re-rank (RapidFuzz, batched two-stage)
                                   │
                       Aggregate to per-code scores
             avg(all rows) × avg(top rows) × √log10(%rows in top pool)
                                   │
          [Optional LLM: final XML re-rank of top codes with reasons]
                                   │
                          Top-k codes to spreadsheet
```

**Key artifacts built once**

* SentenceTransformer embeddings for all SNOMED/RxNorm terms.
* FAISS indices (`HNSW` by default; `IVFPQ` and `Flat` also supported).
* A compact **STY vocabulary** and its embedding matrix for fast STY similarity.

---

## Why it’s novel

* **Multi-signal scoring**: combines four orthogonal signals

  * `direct` (input ↔ code term), `desc` (LLM description ↔ term),
  * `kw` (LLM keywords ↔ term), `sty` (predicted STY ↔ code STY).
* **Large-K pipeline**: searches large candidate pools efficiently, then **fuzzy re-ranks** with a **two-stage batched** RapidFuzz (`ratio` prefilter → `token_set_ratio` refine).
* **Per-code aggregation**: scores *families* of rows (same code, different TTY/strings) with a stability boost based on how many instances survive into the top pool.
* **LLM exchange format is strict XML** for deterministic parsing; LLM is *optional* and fully replaceable with the legacy pipeline.
* **Scalable & resource-aware**: caps FAISS/BLAS threads, sizes the asyncio CPU pool, and bounds concurrent GPU/CPU work with semaphores to avoid oversubscription.

---

## Repo layout

```
.
├─ clean.py                     # main pipeline (build/query/batch + LLM/FAISS/fuzzy)
├─ solution-notebook-demo.ipynb # notebook walkthrough of the solution, simple demo
├─ snomed_all_data.parquet      # SNOMED CT data (provided)
├─ rxnorm_all_data.parquet      # RxNorm data (provided)
├─ Test.xlsx                    # input sheet (sample cases)
├─ Test_with_predictions.xlsx   # (output file)
├─ Column Reference Guide.md    # column dictionary for the parquet files
└─ README.md                    # this file
```

> The screenshot near the top shows an example directory view.

---

## Quickstart

> **Python 3.10+** recommended.

### 1) Install dependencies

**CPU-only (simplest)**

```bash
pip install -U pandas pyarrow openpyxl numpy tqdm \
  sentence-transformers faiss-cpu rapidfuzz llama-cpp-python
```

**GPU (optional, for faster LLM / embeddings)**

```bash
pip install -U vllm bitsandbytes "vllm[flashinfer]" torch --index-url https://download.pytorch.org/whl/cu121
# Ensure a matching CUDA toolkit/driver is present on your machine.
```

> If you’re on a Debian/Ubuntu runtime that needs CUDA libs, see the commands in the next section. For local machines with proper GPU drivers, you usually **don’t** need `apt` steps.

---

## Build the indices

This step embeds all SNOMED/RxNorm strings and creates FAISS indices + STY embeddings.

```bash
python clean.py build \
  --snomed_parquet snomed_all_data.parquet \
  --rxnorm_parquet rxnorm_all_data.parquet \
  --out_dir indices
```

**Optional Hugging Face login** (if a model requires it):

```bash
python clean.py build ... --hf_token YOUR_HF_TOKEN
```

**Expected artifacts in `indices/`**

* `snomed.index.faiss`, `rxnorm.index.faiss`
* `snomed_catalog.parquet`, `rxnorm_catalog.parquet`
* `snomed_vectors.f32`, `rxnorm_vectors.f32` (memmaps)
* `sty_vocab.json`, `sty_embeddings.npy`
* `meta.json` (all settings captured)

**Build time (rough order of magnitude)**

* ~10–60 minutes depending on hardware and index type/size. HNSW is the default and works well.

---

## Run: batch or single query

### A) **Batch** (recommended)

**LLM OFF (pure semantic + fuzzy)** — fastest

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx \
  --out_file out.xlsx \
  --use_llm_clean false --use_llm_rerank false \
  --llm_backend auto \
  --llm_concurrency 200 --rows_concurrency 128
```

**LLM ON (query expansion + final re-rank)** — best for ambiguity

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx \
  --out_file out.xlsx \
  --use_llm_clean true --use_llm_rerank true \
  --llm_backend auto \
  --llm_concurrency 200 --rows_concurrency 128
```

Want the full top-k in your spreadsheet?

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx --out_file out.xlsx \
  --use_llm_clean false --use_llm_rerank false \
  --include_topk true \
  --llm_concurrency 200 --rows_concurrency 128
```

### B) **Single query** (stdout)

```bash
python clean.py query \
  --text "hcv rna" \
  --entity_type "Procedure" \
  --index_dir indices \
  --use_llm_clean false --use_llm_rerank false \
  --llm_backend auto
```

---

## Output format

For each input row (`Input Entity Description`, `Entity Type`) you get:

* **Output Coding System** → `SNOMEDCT_US` or `RXNORM`
* **Output Target Code** → code as **string**
* **Output Target Description** → human-readable label
* **Output Semantic Type (STY)** → e.g., *Disease or Syndrome*, *Pharmacologic Substance*
* **Output Term Type (TTY)** → e.g., `PT`, `SCD`, `BN`, …

With `--include_topk`, additional columns are appended for each rank `k`:
`Output Coding System k`, `Output Target Code k`, `Output Target Description k`, `Output Semantic Type k`, `Output Term Type k`.

See **Column Reference Guide.md** for vocabulary column definitions used inside the engine.

---

## Tuning & performance knobs

**Index**

* `--index {hnsw,ivfpq,flat}` | `--hnsw_m` `--hnsw_efc` `--hnsw_efs` | `--ivf_nlist` `--pq_m` `--pq_nbits`

**LLM (optional)**

* `--llm_backend {auto,vllm,llama-cpp}`
* `--llm_hf_model_id`, `--llm_gguf_path` (auto-selected to Qwen3-4B GGUF on CPU)
* `--llm_concurrency`, `--llm_tp` (tensor parallel for vLLM), `--llm_n_threads` (llama-cpp)
* `--vllm_quantization` (defaults to bitsandbytes)

**Batch concurrency & CPU threads**

* `--rows_concurrency` (how many rows processed concurrently; defaults to LLM concurrency)
* `--cpu_pool` (sized ThreadPool for `asyncio.to_thread` work)
* `--faiss_threads` (FAISS OpenMP threads)
* `--blas_threads` (NumPy/MKL/OpenBLAS cap via threadpoolctl)

**Fuzzy**

* `--fuzzy_workers` (RapidFuzz cdist; `-1` = all cores)

**Scoring weights**

* `--weights_desc` `--weights_kw` `--weights_direct` `--weights_sty` (default `0.30/0.40/0.20/0.10`)

**Legacy rerank switch**

* `--rerank {none,rapidfuzz}` (when LLM is off)

---

## Methodology details

### Embeddings & FAISS

* SentenceTransformer encodes every `STR` in SNOMED/RxNorm (L2-normalized float32).
* FAISS index type is configurable (`HNSW` default).
* Search returns similarities that we map to `[0,1]` by `cos_to_01(x) = clip((x+1)/2)`.

### LLM-assisted **query expansion**

* System prompt constrains responses to **strict XML** with one or more `<candidate>` blocks:

  * `<alternate_keywords>` (brand/colloquial terms included)
  * `<description>` (1–3 sentences)
  * `<possible_semantic_term_types>` (from our STY inventory)
* We **cap to ~5 candidates** and feed these signals into the semantic search.

### Multi-signal scoring (per-row)

For each candidate row we compute:

* **desc**: cosine(description, STR)
* **kw**: max cosine over all alternate keyword vectors
* **direct**: cosine(input text, STR)
* **sty**: STY similarity via a **pre-embedded STY vocabulary** (`sty_vocab`, `sty_embeddings.npy`)

Weighted sum:

```
composite = wd*desc + wk*kw + wr*direct + ws*sty
```

### Fuzzy re-rank (stability to noise)

* Build a **pool** of the best rows (e.g., 500 by composite), then re-rank by **RapidFuzz**:

  * Stage-1: fast `ratio` to prefilter choices for each anchor (query + top alt keywords)
  * Stage-2: `token_set_ratio` on the reduced set
* Implemented as **batched cdist** calls with parallel workers.

### Per-code aggregation

Many vocabularies have multiple rows per code (synonyms, forms, etc.). We:

1. Reconstruct or load vectors for **all rows** of each surviving code.
2. Recompute the components against the best available representations (query/desc/kws).
3. Score the **code** by:

```
final = mean(composite over all rows) *
        mean(composite over top-pool rows for that code) *
        sqrt(log10( percentage_of_code_rows_in_top_pool ))
```

This rewards codes that are **consistently** strong across their variants.

### Optional LLM final re-rank

* We send a compact list: `system | code | STY | TTY | STR` for the top candidates.
* The LLM outputs strict XML `<choice>` blocks with `<code>` + `<reasoning>`.
* Output is intersected with our candidates (we keep order) and filled if needed.

### Resource-awareness

* `configure_cpu_threading()` sets FAISS OMP threads and caps BLAS pools; heavy sections run inside a `threadpool_limits` context.
* Async path sizes a single `ThreadPoolExecutor` and sets `CPU_POOL` env; internal semaphores for embeddings/FAISS use that to avoid oversubscription.

---

## Rules compliance

* **Open-source only**: uses SentenceTransformers, FAISS, RapidFuzz, vLLM/llama-cpp, PyTorch — all open-source.
* **No external/proprietary APIs**: LLMs run locally via vLLM or llama-cpp; Hugging Face downloads public models if you opt in.
* **Outputs include**: system, code, description (+ STY/TTY) exactly as requested.

---

## Deliverables checklist (for evaluators)

* ✅ **Public repo** with runnable code (`clean.py`)
* ✅ **README** (this file) explaining approach, setup, and usage
* ✅ **Run script examples** under “Quickstart / Run”
* ✅ **Produces outputs** with required columns
* ✅ **Handles ambiguity/noise** via LLM expansion + fuzzy re-rank
* ✅ **Complete docs** for dataset columns (see *Column Reference Guide.md*)
* ✅ **Demo options**:

  * `query` for quick console demo
  * `batch` to generate a full **`out.xlsx`** for the judges

---

## Troubleshooting

* **`NameError: nullcontext`**
  Import was omitted in some Python builds. Add:

  ```python
  from contextlib import nullcontext
  ```
* **CUDA/VLLM issues**
  Make sure your CUDA driver matches the PyTorch/cu version. If you’re on a managed Linux box that needs CUDA libs:

  ```bash
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8 libcurand-dev-12-8
  ```

  Or simply run with `--llm_backend llama-cpp` to stay CPU-only.
* **Slow/oversubscribed CPU**
  Try: `--blas_threads 1 --faiss_threads 1 --cpu_pool 8 --rows_concurrency 8`
* **Memory pressure while building**
  Use smaller batch: `--batch 256` and prefer `IVFPQ` index:

  ```bash
  --index ivfpq --ivf_nlist 16384 --pq_m 64 --pq_nbits 8
  ```
* **Fuzzy stage too slow**
  Lower the pool or worker count: `--fuzzy_workers 4` and/or reduce `AdvancedParams.fuzzy_pool` (see code defaults).

---

## Notes on data & licensing

* **SNOMED CT** is licensed; use only within terms permitted for your jurisdiction/institution.
* **RxNorm** is released by the U.S. National Library of Medicine; check their terms.
* This repo assumes you have permission to use the provided parquet files for the hackathon.

---

### Appendix: Example commands (copy/paste)

**Build**

```bash
python clean.py build \
  --snomed_parquet snomed_all_data.parquet \
  --rxnorm_parquet rxnorm_all_data.parquet \
  --out_dir indices
```

**Batch (LLM off)**

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx --out_file out.xlsx \
  --use_llm_clean false --use_llm_rerank false \
  --llm_backend auto \
  --llm_concurrency 200 --rows_concurrency 128
```

**Batch (LLM on)**

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx --out_file out.xlsx \
  --use_llm_clean true --use_llm_rerank true \
  --llm_backend auto \
  --llm_concurrency 200 --rows_concurrency 128
```

**Single query**

```bash
python clean.py query \
  --text "chest xr" \
  --entity_type "Procedure" \
  --index_dir indices \
  --use_llm_clean true --use_llm_rerank true \
  --llm_backend auto
```

---

## Acknowledgements

* FAISS, SentenceTransformers, RapidFuzz, vLLM, llama-cpp, PyTorch, and the maintainers of Qwen models.
* SNOMED CT and RxNorm teams for the vocabularies used in this challenge.

---