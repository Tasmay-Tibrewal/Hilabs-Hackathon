# Clinical Concept Harmonizer — Architecture Deep Dive

> Version: 2025-01 · Target Problem: HiLabs Hackathon 2025 · Repository entrypoint: `clean.py`

This document explains the system in **extreme detail**: goals, data model, build-time/indexing pipeline, query-time pipeline, algorithms, concurrency model, GPU/CPU execution plans, memory management, ranking theory, prompt engineering, configuration knobs, failure handling, performance characteristics, and extensibility. It is designed as a companion to the code in `clean.py`.

---

## 1) Goals, Non‑Goals, and Constraints

### Goals

* **High-accuracy clinical concept mapping** from messy inputs to **RxNorm** (meds) and **SNOMED CT** (everything else).
* **No external paid APIs**; use open-source models only.
* **Scalable** to millions of catalog rows with affordable compute (FAISS + compact embedding model).
* **Latency-friendly** for interactive queries **and** high-throughput for batch.
* **Deterministic fallback** when LLM is disabled (legacy ANN + fuzzy).

### Non‑Goals

* Not a general medical QA system; it maps short clinical strings to codes.
* Not a closed-loop ontology editor; the catalogs are provided as immutable inputs.

### Key Constraints

* Must run on **CPU-only** (using `llama-cpp`) and **GPU** (using `vLLM`).
* **Open-source only**: sentence transformers, faiss, rapidfuzz, vLLM, llama.cpp.
* **Reproducible** artifacts: indices + vector files + meta.

---

## 2) Data Model and Files

### Catalog schema (after preparation)

Each vocabulary (SNOMED, RxNorm) is normalized to the same internal schema:

| Column   | Role                                         |
| -------- | -------------------------------------------- |
| `row_id` | Stable integer id (0..N-1) for FAISS mapping |
| `CODE`   | String code in source vocabulary             |
| `STR`    | Human-readable term (name)                   |
| `CUI`    | UMLS concept id (if provided)                |
| `TTY`    | Term type (PT/SY/SCD/BN/…)                   |
| `STY`    | Semantic type string                         |
| `System` | `SNOMEDCT_US` or `RXNORM`                    |

### Build artifacts written to `--out_dir`

* `snomed_catalog.parquet`, `rxnorm_catalog.parquet`: canon catalogs.
* `snomed.index.faiss`, `rxnorm.index.faiss`: FAISS indices (HNSW/IVFPQ/Flat-IP).
* `snomed_vectors.f32`, `rxnorm_vectors.f32`: memory-mapped normalized embedding matrices (float32).
* `sty_vocab.json` + `sty_embeddings.npy`: vocabulary of STYs present across both catalogs and their embeddings.
* `meta.json`: complete reproducibility metadata (embedding model id, index type/params, paths, dim, etc.).

**Why memory-mapped vectors?**

* Reconstructing vectors from FAISS can be slow/imprecise depending on index type. Persisting the exact normalized embedding used at build time enables fast, vectorized, per-code aggregation later.

---

## 3) Build-Time Pipeline (Indexer)

**Entry:** `python clean.py build ...`

### Steps

1. **Load Parquet catalogs** (SNOMED, RxNorm) and call `prepare_catalog()` to project onto the common schema and assign `row_id`.
2. **Load Sentence Embedding model** (default: `google/embeddinggemma-300m`).
3. **Embed all `STR` values** in batches with GPU OOM backoff.
4. **Construct FAISS index** per system:

   * **HNSW Flat-IP**: default. Configurable `M`, `efConstruction`, `efSearch`.
   * **IVFPQ**: trains on a sample; lower memory, fast search at large scale.
   * **Flat-IP**: exact but memory/latency heavy.
5. **Add vectors with IDs** (`row_id`) and persist index files.
6. **Persist memory-mapped matrices** of the same vectors (`*.f32`).
7. **Embed STY vocabulary** once and store `sty_embeddings.npy` for quick semantic-type similarity.
8. **Write `meta.json`** to tie everything together.

### Similarity Metric

All embeddings are L2-normalized; we use **inner product** in FAISS which equals cosine similarity. We convert to [0,1] via `cos_to_01(x) = clip(0.5*(x+1), 0, 1)`.

---

## 4) Query-Time Pipeline — Two Modes

**Entry:** `python clean.py query ...` (single) or `batch` (multi-row).

We implement two matchers:

1. **Legacy ANN + Fuzzy** (**no LLM**): direct semantic search with optional RapidFuzz rerank; system preference (RxNorm for meds else SNOMED); thresholds and margins for switching.
2. **Advanced LLM-Assisted Pipeline** (**default**): multi-signal, multi-candidate reasoning with optional final LLM *code-level* rerank.

### 4.1 Legacy Pipeline (deterministic, low-cost)

* Embed query once; FAISS-search both indices.
* Optionally rerank each system’s topK using `rapidfuzz.token_set_ratio` against the original query.
* Pick **preferred system** (`choose_system(entity_type)`), but **switch** if alternative beats it by `--alt_margin` and passes `--min_score`.
* Return topK rows from the chosen system.

**When to use:** CPU-only, strict determinism, or when LLM is not desired.

### 4.2 Advanced LLM-Assisted Pipeline (default)

#### High-level idea

Use the LLM to **expand** the user input into multiple plausible **candidates** (synonyms/brands/abbreviations + short description + likely STYs). Then score codes using a **weighted mixture of signals** derived from description/keywords/direct match/STY similarity. Finally, **fuzzy re-rank**, **aggregate to per-code scores**, and optionally apply an **LLM reranker** on a small shortlist.

#### Signal Definitions

Let `q` be the raw query. For each catalog row `r` with text `STR_r` and STY `STY_r` we estimate:

* **Direct semantic**: `S_dir(r) = cos01(emb(q) · emb(STR_r))`
* **Description semantic** (from LLM): `S_desc(r) = max_c cos01(emb(desc_c) · emb(STR_r))`
* **Alt-keyword semantic**: `S_kw(r) = max_kw cos01(emb(kw) · emb(STR_r))`
* **STY compatibility**: `S_sty(r) = max_{sty_pred ∈ STY_pred} cos01(emb(sty_pred) · emb(STY_r))`

> `cos01(x) = clip(0.5*(x+1), 0, 1)` converts cosine to [0,1].

Then, the **row-level composite**

[
C(r) = w_{desc}·S_{desc}(r) + w_{kw}·S_{kw}(r) + w_{direct}·S_{dir}(r) + w_{sty}·S_{sty}(r)
]

with defaults `w_desc=0.30, w_kw=0.40, w_direct=0.20, w_sty=0.10` (tunable CLI flags).

#### Candidate Expansion (LLM)

* **System prompt** (`EXPAND_SYSTEM_PROMPT`) constrains output to **pure XML** of `<candidate>` blocks.
* For each candidate:

  * `<alternate_keywords>`: 2–10 short aliases/brands/abbreviations.
  * `<description>`: 1–3 sentence medical explanation.
  * `<possible_semantic_term_types>`: pick from the **provided STY list** only.
* Parser `extract_xml_candidates()` is defensive: handles missing fields, odd whitespace, single-block fallbacks.

#### Coarse Retrieval (FAISS)

For each signal family we perform **vector search** against **both** indices:

* Once for the **direct** query (`q`).
* Once per **candidate description**.
* **Batched** over all **alternate keywords** (much faster than per-kw loops).

Each hit contributes to its row’s `score_components` via the `cos01`-normalized score.

#### Fuzzy Re-rank (RapidFuzz)

We take the top **`fuzzy_pool`** rows by composite (default 500), then add a lexical check using **two-stage** RapidFuzz:

1. **Prefilter** with `ratio` on normalized strings for each anchor (query + best K keywords). Keep top-N indices union.
2. **Refine** candidates with `token_set_ratio` in parallel (`cdist`, `workers=-1` → all cores).

The `fuzzy` score is **not** blended with the composite numerically here; we simply **sort** by fuzzy for a **stability** pass and truncate to `top_pool_after_fuzzy` (default 250). This protects against semantic drift from embeddings when inputs are very short.

#### Per-Code Aggregation and Boost

Rows are bucketed by `(System, CODE)`. For each code we compute:

* `avg_all` = mean of **recomputed** composites across **all** rows of the code (using **memory-mapped vectors** for speed). Recompute uses the **best available** signals for the query (desc, kw, direct, STY).
* `avg_500` = mean of composites across only the rows that survived the fuzzy pool (`rows_top`).
* Let `p = 100 × |rows_top| / |rows_all|`. Compute a conservative **frequency boost**:

[
\text{boost}(p) = \sqrt{\log_{10}(\max(1.0001, p))}
]

Final **code score**:

[
F(\text{code}) = \text{avg_all} × \text{avg_500} × \text{boost}(p)
]

This favors codes that are **consistently strong** across their terms and have **good representation** in the top pool, mitigating outliers.

#### Final LLM Rerank (Optional)

* Build a small candidate list (default 30 codes) with **representative string** per code (prefers PT/FN/SCD via a rank map).
* Provide the LLM with:

  * Original query + entity type.
  * **Expanded understanding** summary (all candidates, their keywords and STYs).
  * Compact list `system | code | STY | TTY | STR` for each candidate.
* The LLM returns a **pure XML list** of `<choice>` with **only codes** and a short reason. We re-map codes back to best `(System, Code)` pair and **preserve at most `final_output_k`**.
* If LLM returns fewer items, we **fill** from the highest-scoring remaining codes.

---

## 5) LLM Backends and Auto‑Selection

Two interchangeable backends via `LLMConfig`:

* **vLLM** (GPU): loads HF weights (defaults to **Qwen3 4B** for smaller VRAM; **GPT‑OSS‑20B** otherwise) with optional quantization (`bitsandbytes`, `awq`, `gptq`, `mxfp4`).
* **llama-cpp** (CPU): loads **GGUF** (defaults to Unsloth Qwen3 4B Q4_K_XL). Threaded generation via `ThreadPoolExecutor`.

**Auto policy** (`pick_llm_backend_and_model`):

* If **no GPU** → `llama-cpp` with Qwen3 4B GGUF (downloaded from HF if needed).
* If GPU exists but **<22 GB per GPU** → **vLLM + Qwen3 4B**.
* Else → **vLLM + GPT‑OSS‑20B**.

LLM is used in two places only:

1. **Expansion** (keyword + desc + STY hints) — many parallel short prompts.
2. **Final rerank** — one prompt per query, small payload.

Both are **bounded** and run concurrently.

---

## 6) Concurrency Model and Threading

### Async Orchestration

* Batch mode creates an **async event loop** and limits shared resources using semaphores:

  * `_embed_sem`: throttles GPU-heavy embedding jobs (defaults to `2×num_gpus`, capped by `CPU_POOL`).
  * `_faiss_sem`: throttles FAISS searches (≈ `2×_embed_sem`).
* A configurable **row concurrency** (`--rows_concurrency`) controls how many records are processed in parallel at the pipeline level.

### LLM Concurrency

* **vLLM**: uses its own internal scheduler; we set `max_num_seqs` and enable **prefix caching** to exploit identical system prompts.
* **llama-cpp**: the engine is not thread-safe for concurrent calls; we serialize each chat completion via an **async lock**, but run the blocking call in a **thread pool** sized by `LLMConfig.concurrency`.

### CPU Thread Control (`configure_cpu_threading`)

* Sets FAISS **OpenMP** threads via `faiss.omp_set_num_threads`.
* Caps BLAS backends via `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.
* Optionally uses `threadpool_limits` context to bound NumPy/Scipy and friends during heavy sections.

### Why two levels of concurrency?

* **Pipeline level** (rows): hides per-row variability and IO.
* **Component level** (emb/FAISS/LLM): prevents head-of-line blocking and GPU OOM while keeping hardware busy.

---

## 7) Memory Management and OOM Resilience

* Embedding uses **backoff batching** on CUDA OOM: halving batch size until it fits (down to 1).
* `clear_cuda_memory()` aggressively frees caches and triggers GC between phases.
* `model_roundtrip_cpu_gpu()` can be used as a VRAM defragmentation nudge.
* FAISS indices live on **CPU RAM**; vectors are normalized **once** and stored on disk as **memmaps** for later **vectorized** scoring without rebuilding.

---

## 8) Prompt Engineering and XML Parsing

### Expansion Prompt

* Strongly constrained to **ASCII XML** with only three tags. We provide the **finite list of allowed STYs** so the model cannot invent new labels.
* We encourage **multiple candidates** when the input is ambiguous (e.g., "XR chest" → imaging vs. order).

### Rerank Prompt

* Positions the LLM as an adjudicator over a **shortlist of standardized codes**. The model does **not invent new codes**—it only ranks provided ones.
* Output again is pure XML to simplify parsing.

### Parsers

* `extract_xml_candidates()`/`extract_xml_choices()` use robust regex splits with graceful fallbacks (single-block or partial field missing → skip/empty lists).

---

## 9) Algorithmic Complexity & Performance Notes

Let `N_sno`, `N_rx` be catalog sizes, `d` embedding dim.

* **Build time**: `O((N_sno+N_rx) × cost(embed)) + O(N log N)` (index construction), dominated by embedding.
* **Query time** (per row):

  * Embedding: `O(#texts × d)`; typically small (query + 0–10 KWs + ≤1 desc).
  * FAISS search: `O(log N)` average for HNSW; `O(d × efSearch)` tuned by `efSearch`.
  * Fuzzy cdist: `O(A × M)` where `A` anchors (≤9) and `M` reduced set (≤prefilter). Leveraging C++ + multithread keeps it small.
  * Aggregation uses memmap **matrix multiplications** for each code (vectorized in NumPy): highly efficient.

**Empirical**

* Build: ~11 min on 1×H100, 30–60 min on T4-class GPUs; ~10GB disk for artifacts (depends on index type).
* Batch LLM-off: seconds to sub-minute for typical test sheets.
* Batch LLM-on: depends on GPU and concurrency settings; designed to scale linearly with hardware.

---

## 10) Configuration Knobs (Cheatsheet)

### Build

* `--model`: sentence embedding model id.
* `--index`: `hnsw|ivfpq|flat` and their params (`--hnsw_m`, `--hnsw_efc`, `--hnsw_efs`, `--ivf_nlist`, `--pq_m`, `--pq_nbits`).
* `--batch`: embedding batch size.

### Query/Batch (Core)

* `--use_llm_clean`, `--use_llm_rerank`: toggle LLM stages.
* `--topk`: final outputs per row.
* `--rerank`: `none|rapidfuzz` legacy path option.
* `--min_score`, `--alt_margin`: legacy system-switch policy.

### Weights and Pools

* `--weights_desc`, `--weights_kw`, `--weights_direct`, `--weights_sty`.
* Advanced params (compiled): `semantic_top_desc`, `semantic_top_kw_each`, `fuzzy_pool`, `top_pool_after_fuzzy`, `final_llm_candidates`.

### LLM Backend

* `--llm_backend`: `auto|vllm|llama-cpp`.
* `--llm_hf_model_id`, `--llm_gguf_path`.
* Generation: `--llm_max_new_tokens`, `--llm_temperature`, `--llm_top_p`.
* Scaling: `--llm_concurrency`, `--llm_tp`, `--llm_n_threads`, `--vllm_quantization`.

### CPU/Threads

* `--blas_threads`, `--faiss_threads`, `--cpu_pool`, `--fuzzy_workers`.
* Batch: `--rows_concurrency`.

---

## 11) Failure Modes & Mitigations

| Risk                   | Symptom                           | Mitigation                                                                        |
| ---------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| CUDA OOM               | Embedding crashes                 | backoff batches; clear CUDA cache; lower `--batch`; use CPU or smaller model      |
| vLLM VRAM limits       | Engine fails to initialize        | auto-select Qwen4B; set `--vllm_quantization` to `mxfp4`/`awq`; reduce `--llm_tp` |
| llama-cpp contention   | Random slowdowns                  | serialize access (`_lock`); adjust `--llm_concurrency`; increase `--n_threads`    |
| Over-threading         | CPU pegged with little throughput | set `--blas_threads`, `--faiss_threads`, `--cpu_pool` sanely                      |
| Bad LLM XML            | No candidates/choices parsed      | robust regex + fallbacks; pipeline still returns **non-LLM** results              |
| Ambiguous short inputs | Wrong concept                     | rely on multi-candidate expansion + fuzzy; consider raising `w_kw` temporarily    |

---

## 12) Security, Privacy, and Reproducibility

* **Data never leaves the machine**: no remote API calls; only HF hub downloads of **open** model weights (optional token for rate limits).
* **Reproducibility**: `meta.json` captures model id + dims + index params + filenames. Artifacts are pure files—no hidden state.
* **Determinism**: Legacy path is deterministic; LLM path can vary slightly across seeds/hardware; final lists are constrained to existing codes.

---

## 13) Extensibility

* **Add a new vocabulary**: prepare same 7-column schema; train/embed; add new FAISS index + memmap; extend `choose_system()` policy.
* **New signal**: add term-level signal and weight in `AdvancedWeights` + accumulation in `score_components`.
* **Alternate embedding model**: update `--model` at build; re-build indices and STY embeddings.
* **Richer rerank**: swap RapidFuzz scorer or combine fuzzy numerically in the composite if desired.

---

## 14) Worked Example (End-to-End)

Input row:

```
Input Entity Description: "chest xr"
Entity Type: "Procedure"
```

1. **LLM expansion** → candidates: {"chest x-ray", "cxr", "radiograph chest"}, desc ≈ “plain radiograph of the chest”, STY hints "Diagnostic Procedure".
2. **Semantic searches**: direct on "chest xr"; desc search; batched kw searches.
3. **Row composites** built across both indices.
4. **Fuzzy** stage boosts strings containing both tokens {"chest", "x-ray"}.
5. **Aggregate** per code: many SNOMED rows share X-ray procedures; `avg_all` and `avg_500` select the consistent code; frequency boost favors codes with multiple supporting synonyms.
6. **LLM rerank** (optional): chooses the best procedure code and explains briefly.

---

## 15) Pseudocode (Advanced Matcher)

```text
exp_cands = LLM.expand(q, entity_type)
q_vec = emb(q)
for cand in exp_cands:
  d_vec = emb(cand.description)
  kw_vecs = emb(cand.keywords)
  for sys in {SNOMED,RXNORM}:
    add_scores(search(sys, d_vec), "desc")
    add_scores_batch(search(sys, kw_vecs), "kw")
  add_scores(search(SNOMED, q_vec), "direct")
  add_scores(search(RXNORM, q_vec), "direct")
  apply_sty_map(predicted_stys)

rows = top_by_composite(fuzzy_pool)
rows = fuzzy_resort_and_truncate(rows, top_pool_after_fuzzy)

codes = group_by_code(rows)
for code in codes:
  all_vecs = memmap_vectors(code)
  recompute composites vs {q_vec, d_vec, kw_vecs, sty_map}
  avg_all, avg_500, p = reduce(code)
  score = avg_all * avg_500 * sqrt(log10(max(1.0001,p)))

shortlist = topN(codes)
if use_llm_rerank:
  final = LLM.rerank(shortlist, context)
else:
  final = shortlist[:K]
```

---

## 16) Known Limitations & Future Work

* **Unit handling**: current system treats dosage/quantities lexically; a dose parser could improve RxNorm precision.
* **Negation/qualifiers**: phrases like “rule out pneumonia” may map too strongly to the disease; adding a **clinical negation** detector would help.
* **Context window**: single-field input; future: multi-column fusion (age/sex/lab units) for disambiguation.
* **Learning-to-rank**: the score weights are fixed; we could learn them from labeled data.
* **Cache**: response caching (query→result) for repeats in batch runs.

---

## 17) Practical Tuning Recipes

* **CPU-only laptop**: `--use_llm_clean false --use_llm_rerank false --rerank rapidfuzz --blas_threads 4 --faiss_threads 4`.
* **Single 8–12GB GPU**: `--llm_backend auto --vllm_quantization mxfp4 --llm_concurrency 64 --rows_concurrency 64`.
* **Accuracy-first**: increase `w_kw` and `w_desc`; ensure LLM stages enabled; raise `final_llm_candidates` to 50.
* **Throughput-first**: set `use_llm_rerank=false`; keep expansion on; reduce `semantic_top_*` and `fuzzy_pool`.

---

## 18) CLI Quick Reference

### Build

```bash
python clean.py build \
  --snomed_parquet snomed_all_data.parquet \
  --rxnorm_parquet rxnorm_all_data.parquet \
  --out_dir indices \
  --model google/embeddinggemma-300m
```

### Batch (LLM on)

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx --out_file out.xlsx \
  --use_llm_clean true --use_llm_rerank true \
  --llm_backend auto \
  --llm_concurrency 200 --rows_concurrency 128
```

### Batch (LLM off)

```bash
python clean.py batch \
  --index_dir indices \
  --in_file Test.xlsx --out_file out.xlsx \
  --use_llm_clean false --use_llm_rerank false \
  --rerank rapidfuzz
```

### Single Query

```bash
python clean.py query \
  --index_dir indices \
  --text "hcv rna" \
  --entity_type "Procedure" \
  --use_llm_clean false --use_llm_rerank false
```

---

## 19) File/Module Map

* **`clean.py`** — all logic: build, load, query, batch; FAISS, embeddings, LLM adapters, fuzzy, scoring, CLI.
* **Artifacts in `indices/`** — indices, memmaps, STY embeddings, meta.
* **`Test.xlsx`** — input; **`out.xlsx`** — output with columns: system, code, description, STY, TTY (+ optional topK columns).

---

## 20) Summary

This architecture **blends** fast vector search, lexical sanity checks, semantic-type priors, and *constrained* LLM reasoning. It is:

* **Accurate** (multi-signal evidence + optional LLM adjudication),
* **Scalable** (FAISS + batched embeddings + memmaps),
* **Portable** (GPU/CPU), and
* **Reproducible** (file artifacts + meta).

The system is purpose-built for the Hackathon’s harmonization task while remaining extensible for real-world production use.