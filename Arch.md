# Architecture & Internals — Clinical Concept Harmonizer

> Extreme-detail documentation of the design, algorithms, and execution model that power `clean.py`.

---

## 0) Goals, Non‑Goals, and Design Constraints

**Goals**
- Normalize raw clinical strings to **RxNorm** (medications) and **SNOMED CT** (diagnoses, procedures, labs).
- Be **robust** to abbreviations, spelling noise, and incomplete phrases.
- Run **completely offline** with **open-source** components.
- Scale to millions of terminology rows with **fast semantic search** and **careful concurrency**.
- Offer an **optional LLM assist** that increases recall/precision without locking us into a proprietary API.

**Non‑Goals**
- No training/fine-tuning loop in this repo. We rely on public embedding models + heuristic scoring.
- No external inference APIs. All LLM inference runs locally (vLLM or llama‑cpp).

**Constraints**
- Works on **CPU** (llama‑cpp) and **GPU** (vLLM). Auto-selects a backend by VRAM.
- Deterministic I/O and portable artifacts (FAISS indices, memmaps, STY vectors).

---

## 1) End‑to‑End Dataflow

```
                  ┌────────────────────────────────────────────────────────────────────┐
                  │               One‑time Build (clean.py build)                      │
Raw Parquets ──►  │ 1) prepare_catalog()    → canonical schema                         │
 (SNOMED/RxNorm)  │ 2) embed_texts()        → STR embeddings (float32, normalized)    │
                  │ 3) build_* FAISS        → HNSW/IVFPQ/Flat wrapped in IDMap2       │
                  │ 4) save memmaps         → snomed_vectors.f32 / rxnorm_vectors.f32 │
                  │ 5) STY vocab + emb      → sty_vocab.json + sty_embeddings.npy      │
                  │ 6) meta.json            → reproducibility record                   │
                  └────────────────────────────────────────────────────────────────────┘

                                  (indices/, memmaps/, sty_*.{json,npy})

                               ┌─────────────────────────────────────────────┐
                               │             Query / Batch Run               │
Input Row  ─► LLM (optional) ─►│ 1) Query expansion (XML): alt KWs, desc,    │
(text,type)                    │    candidate STYs                            │
                               │ 2) Semantic search (FAISS) over both systems│
                               │    • direct(query) • description • keywords │
                               │ 3) Per-row weighted score (desc,kw,dir,sty) │
                               │ 4) Pool top N rows → RapidFuzz two‑stage    │
                               │    fuzzy re‑rank                             │
                               │ 5) Aggregate by code (avg_all * avg_top     │
                               │    * stability boost)                       │
                               │ 6) LLM (optional) final re‑rank via XML     │
                               │ 7) Emit top‑k: System/CODE/STR/STY/TTY      │
                               └─────────────────────────────────────────────┘
```

---

## 2) Core Files & Artifacts

**Runtime artifacts (under `indices/`):**
- `snomed.index.faiss`, `rxnorm.index.faiss` — ANN indices (`HNSW` default). Wrapped with `IndexIDMap2` to address by `row_id`.
- `snomed_catalog.parquet`, `rxnorm_catalog.parquet` — normalized catalogs: `[row_id, CODE, STR, CUI, TTY, STY, System]`.
- `snomed_vectors.f32`, `rxnorm_vectors.f32` — memory‑mapped, L2‑normalized float32 matrices aligned to `row_id`.
- `sty_vocab.json` + `sty_embeddings.npy` — the **universe of STY strings** + their embeddings for STY similarity.
- `meta.json` — captures model name, index type, dimensions, and filenames for reproducibility.

**Key modules in `clean.py`:**
- **Embedding**: `load_model`, `embed_texts` (OOM‑aware with batch backoff), plus async wrappers.
- **FAISS**: builders (`build_hnsw`, `build_ivfpq`, `build_flat`), `faiss_search`, and async wrappers.
- **LLM Backends**:
  - `LLMConfig`, `LLMClient` (sync) and `AsyncLLMClient` (async stream) with auto‑backend selection via `pick_llm_backend_and_model`.
  - Backends: **vLLM** (GPU) with quantization (default bitsandbytes) and **llama‑cpp** (CPU) with GGUF.
- **Prompting**:
  - Expansion (`EXPAND_*`) → candidates in XML.
  - Re‑rank (`RERANK_*`) → `<choice>` XML blocks.
  - Parsers: `extract_xml_candidates`, `extract_xml_choices` (regex‑tolerant, minimal).
- **Pipelines**:
  - Advanced (LLM optional) single query: `advanced_match_one`, async: `advanced_match_one_async`.
  - Legacy (non‑LLM) selection: cosine + optional RapidFuzz rerank.
  - CLI entry points: `build`, `query`, `batch`.

---

## 3) Build Pipeline (One‑time)

**Input**: `snomed_all_data.parquet`, `rxnorm_all_data.parquet`  
**Output**: `indices/` bundle.

Steps:
1. **Canonicalize** with `prepare_catalog(df, system)`  
   - Keep columns `["CUI","System","TTY","CODE","STR","STY"]`; add numeric `row_id` = 0..N-1.
2. **Model**: `SentenceTransformer(model_name)` from `meta["model"]` (default: `google/embeddinggemma-300m`).  
3. **Vectorization**:  
   - Batch encode every `STR`.  
   - Write vectors into memmap files `*.f32` by `row_id`.  
   - **Normalization**: embeddings are L2 normalized, ensuring cosine = inner product (IP).
4. **FAISS index**:  
   - `HNSW` → `IndexHNSWFlat(dim, M, metric=IP)`, set `efConstruction` (trainless).  
   - `IVFPQ` → train on a large sample; parameters `nlist, pq_m, pq_nbits`.  
   - `Flat` → brute force (baseline).  
   - Always wrap with `IndexIDMap2` so we can refer to `row_id` as the FAISS ID.
5. **STY vocabulary**:  
   - Union of `STY` from both vocabularies → embed → save `sty_embeddings.npy` and `sty_vocab.json`.
6. **Meta**:  
   - Persist model/index/paths into `meta.json`.

**Threading**:  
- `configure_cpu_threading()` provides FAISS OpenMP thread control and caps BLAS pools. Heavier build sections run under `threadpool_limits` if available.

---

## 4) Query Pipeline (Advanced, LLM‑optional)

Below is the **core algorithm** implemented by `advanced_match_one` / `advanced_match_one_async`.

### 4.1 Inputs
- `query`: raw string (e.g., `"chest xr"`)
- `entity_type`: free‑text label used for user guidance and system preference (RxNorm for meds; SNOMED otherwise)
- `bundle`: loaded via `load_bundle(index_dir)` (FAISS, catalogs, memmaps, STY objects, model)
- Flags: `use_llm_clean`, `use_llm_rerank` and `AdvancedWeights/Params`

### 4.2 LLM Query Expansion (optional)
- Prompted with a strict XML format:
  - `<candidate>` blocks, each with:
    - `<alternate_keywords>` (brand/common & scientific names)
    - `<description>` (1–3 sentences)
    - `<possible_semantic_term_types>` (values strictly from our STY inventory)
- Parser yields up to ~5 candidates.

### 4.3 Compute Representations
- `q_vec = embed(query)`
- For each candidate:
  - `desc_vec = embed(description)` if present
  - `kw_vecs = embed(each alternate keyword)` (batched)
  - `sty_map = precompute_sty_scores_map(pred_stys, sty_vocab, sty_emb)`  
    Produces **fast lookup**: `candidate_STY → [0,1]` based on cosine to any predicted STY.

### 4.4 Vector Search (both systems)
We run FAISS searches per signal type:
- **direct**: `search(q_vec, topK_direct)`
- **desc**: `search(desc_vec, topK_desc)` if description exists
- **kw**: for each `kw_vec` → `search(..., topK_kw_each)` (batched)
Each hit gets a **cosine‑to‑[0,1] score** and is placed in a `row_scores` dict keyed by `(system, row_id)`; we keep the **max** score from each signal seen so far.

### 4.5 STY Scoring
- For every row seen, look up its row `STY`, score from `sty_map` (already in `[0,1]`), keep max across candidates.

### 4.6 Weighted Composite
For each row we compute:
```
desc01  = cos_to_01(desc_cos)
kw01    = cos_to_01(max_kw_cos)
dir01   = cos_to_01(direct_cos)
sty01   = sty_map[row.STY] or 0
composite = w_desc*desc01 + w_kw*kw01 + w_direct*dir01 + w_sty*sty01
```
Defaults: `w_desc=0.30, w_kw=0.40, w_direct=0.20, w_sty=0.10`.

### 4.7 Fuzzy Re‑rank (two‑stage, batched)
- Take top `fuzzy_pool` rows by `composite`.  
- Build **anchors** = normalized **query** + up to `kw_limit` alternate keywords.  
- Stage‑1 prefilter: for each anchor compute `process.cdist([anchor], choices, scorer=ratio)` to keep only the best `prefilter` indices.  
- Stage‑2 refine: `token_set_ratio` via `cdist` on the reduced set.  
- Assign each row its **max** fuzzy score in `[0,1]`, sort by this score, then keep `top_pool_after_fuzzy` rows.

### 4.8 Aggregate to Per‑Code Scores
Codes may have numerous rows (synonyms, forms). For each surviving **code**:
- Gather **all rows** of that code from the catalog.
- **Recompute** components against:
  - `q_vec` (direct)
  - best `desc_vec` among the candidates (if any)
  - all `kw_vecs` (take **max** over keywords)
  - `sty_map_all` (union of predicted STYs from all candidates)
- Vectorized compute using **memmaps** (no FAISS reconstruct needed).
- Let:
  - `avg_all` = mean(composite over **all** rows for this code)
  - `avg_500` = mean(composite over the code’s rows that survived into the **top pool**)
  - `%in_top` = 100 * (#rows_in_top_pool / total_rows_of_code)

Final per‑code score:
```
boost = sqrt(log10(max(1.0001, %in_top)))
final = avg_all * avg_500 * boost
```
This rewards **breadth and consistency**: codes that appear frequently and strongly in the initial pool surface higher.

### 4.9 Optional LLM Final Re‑rank
- Build a compact blob: `system | code | STY | TTY | STR` for top‑N codes.
- Prompt the LLM to return XML `<choice>` items (`<code>…</code>`, `<reasoning>…</reasoning>`).
- Keep in LLM order **only** if the code existed in our candidate set; fill remaining slots from the original ranking.

### 4.10 Emit
- For batch mode, write top‑1 (and optionally top‑k) columns to the spreadsheet:
  - `Output Coding System`, `Output Target Code`, `Output Target Description`,
  - `Output Semantic Type (STY)`, `Output Term Type (TTY)`.

---

## 5) Legacy (Non‑LLM) Pipeline

- Embed the **query** only.
- FAISS search on SNOMED and RxNorm. Optionally rerank each list with RapidFuzz `token_set_ratio`.
- Choose the **preferred system** (RxNorm for `Entity Type == Medicine`, else SNOMED) unless the **alternate** is **≥ min_score** and **significantly better** (`alt_margin`).
- Output top‑k.

This path is simple and fast, and remains the fallback if LLM is disabled.

---

## 6) Concurrency & Resource Management

**Threading knobs** (CLI):
- `--faiss_threads`: sets FAISS OpenMP threads via `faiss.omp_set_num_threads`.
- `--blas_threads`: caps NumPy/BLAS pools by setting env vars and using `threadpool_limits` context.
- `--cpu_pool`: sizes the `ThreadPoolExecutor` for `asyncio.to_thread` (embedding, FAISS calls).

**Semaphores**:
- `_embed_sem` and `_faiss_sem` limit concurrent GPU/CPU calls. Defaults scale with number of GPUs; also consider `CPU_POOL` env to avoid oversubscription.

**LLM concurrency**:
- `AsyncLLMClient` uses a `BoundedSemaphore` with `concurrency = default_concurrency()` unless overridden.

**CUDA hygiene**:
- `clear_cuda_memory()` before/after heavy phases.
- `model_roundtrip_cpu_gpu()` nudges allocator fragmentation down when needed.

---

## 7) LLM Backends & Prompts

**Auto selection** (`pick_llm_backend_and_model`):
- **No GPU** → `llama-cpp` with Qwen3‑4B GGUF (`UNSLOTH_GGUF_QWEN_FILE`) via HF hub download.
- **GPU present**:
  - If **< 22 GB VRAM per GPU** → vLLM with **Qwen3‑4B** (bnb).
  - Else prefer larger **OSS‑20B** weights (bnb).

**Sampling defaults**: `max_new_tokens=512, temperature=0.1, top_p=0.9`.

**Prompts** (strict machine‑parsable XML):
- **EXPAND**: produce `n` `<candidate>` blocks: keywords, description, possible STYs (must come from our `sty_vocab`).  
- **RERANK**: given (input, expanded summary, up to 50 candidates), emit up to `k` `<choice>` items with code + short reasoning.

**Parsers** are regex‑tolerant and fail‑soft (empty results degrade gracefully).

---

## 8) Scoring Details & Rationale

**Cosine to [0,1] mapping**:  
`cos_to_01(x) = clip((x + 1)/2, 0, 1)` — makes weights comparable and intuitive.

**Weights** (empirical defaults):  
- `desc` (0.30): semantic alignment to the **meaning**.
- `kw` (0.40): capture **aliases/brands/colloquialisms**.
- `direct` (0.20): strong when input is already near canonical.
- `sty` (0.10): provides **type coherence** without over‑constraining.

**Fuzzy two‑stage**:  
- Stage‑1 (`ratio`) quickly screens to small candidate sets per anchor.  
- Stage‑2 (`token_set_ratio`) is stronger but costlier; we run it only on a reduced set.  
- Parallelized via RapidFuzz `cdist` with `workers` argument.

**Per‑code aggregation**:  
- Avoids picking a single “lucky” string. Codes with **many strong synonyms/forms** win naturally.  
- The **stability boost** uses `sqrt(log10(%in_top))` to modestly reward breadth without allowing explosion.

---

## 9) CLI Surface (Most Relevant Flags)

| Area | Flag(s) | What it does |
|---|---|---|
| Build | `--index {hnsw,ivfpq,flat}` | Choose ANN type |
|  | `--hnsw_m --hnsw_efc --hnsw_efs` | HNSW params (graph degree, construction & search ef) |
|  | `--ivf_nlist --pq_m --pq_nbits` | IVFPQ params |
|  | `--batch` | Embedding batch size |
| LLM | `--use_llm_clean --use_llm_rerank` | Toggle expansion + final rerank |
|  | `--llm_backend {auto,vllm,llama-cpp}` | Select backend or let it auto-decide |
|  | `--llm_concurrency --llm_tp --llm_n_threads` | Throughput controls |
|  | `--vllm_quantization` | e.g., `bitsandbytes`, `mxfp4`, `awq`, `gptq` |
| Perf | `--faiss_threads --blas_threads --cpu_pool` | CPU thread controls |
| Fuzzy | `--fuzzy_workers` | RapidFuzz cdist workers |
| Output | `--include_topk` | Write full top‑k columns to sheet |

---

## 10) Complexity & Performance Notes

Let `N` be the number of concepts per system; `K` the candidate pool sizes.

- **FAISS search**: sub‑linear (HNSW/IVFPQ). Empirically ~ms per query vector per system.  
- **Keyword expansion** multiplies the number of query vectors by (1 + #Kws + #Desc). We bound this to keep runtime predictable.  
- **Fuzzy**: costs are controlled by `prefilter` and `top_pool_after_fuzzy`. The two‑stage design avoids O(N×M) brute force.  
- **Aggregation**: per‑code is vectorized over memmaps; no FAISS reconstruct is required.

Throughput scales with:
- LLM concurrency (if enabled), FAISS OMP threads, and the `cpu_pool` size.
- Avoid oversubscription by tuning `--blas_threads`, `--faiss_threads`, and semaphores.

---

## 11) Failure Modes & Resilience

- **LLM returns malformed XML** → regex fallbacks keep at least 1 candidate; otherwise pipeline continues with legacy signals.  
- **OutOfMemory during embedding** → batch backoff halves the batch size until 1, then raises.  
- **FAISS not trained (IVFPQ)** → `train_ivfpq_if_needed` runs during build with a large sample.  
- **No matches above threshold** (legacy path) → returns empty for that system; selector may fallback to alt system or report “no match ≥ min_score”.  
- **Excel/CSV I/O** → supports `.xlsx/.xls/.csv/.tsv` with correct dtype handling.

---

## 12) Extensibility Roadmap

- **Cross‑encoders** or reranking transformers after FAISS to sharpen semantic precision.  
- **Domain dictionaries**: add curated abbreviation→expansion or lab panels.  
- **Multi‑vocab fusion**: e.g., add **LOINC** for labs; extend `prepare_catalog` to new SABs.  
- **Learned weight calibration**: fit `w_desc/w_kw/w_direct/w_sty` on a validation set.  
- **Caching**: memoize embeddings for repeated tokens and common keywords.  
- **Explainability**: persist per‑row component scores to aid audit/tracing.

---

## 13) Security & Privacy

- No external calls required; all inference is local.  
- Avoid logging PHI; the code prints **only** model/system info and progress bars by default.

---

## 14) Glossary

- **STR**: String/term name in the source vocabulary.  
- **TTY**: Term Type (e.g., `PT`, `SCD`, `BN`).  
- **STY**: Semantic Type (higher‑level category, e.g., *Disease or Syndrome*).  
- **CUI**: Concept Unique Identifier (UMLS concept).  
- **HNSW**: Hierarchical Navigable Small World graph index.  
- **IVFPQ**: Inverted File with Product Quantization (compressed ANN index).

---

## 15) Minimal Pseudocode (Advanced Pipeline)

```python
def match(query, entity_type):
    # 1) expand
    cands = LLM.expand(query, entity_type) if use_llm_clean else [empty_cand]

    # 2) embed anchors
    q_vec = embed(query)
    desc_vecs = [embed(c.desc) for c in cands if c.desc]
    kw_vecs   = embed(all_keywords(cands))
    sty_map   = precompute_sty_scores_map(cands_STYs, sty_vocab, sty_emb)

    # 3) FAISS searches
    rows = defaultdict(RowScores)
    for vec in [q_vec] + desc_vecs + kw_vecs:
        for sys in [SNOMED, RXNORM]:
            hits = faiss.search(sys.index, vec, K)
            for h in hits:
                rows[(sys, h.row_id)].update_component(vec_type, cos_to_01(h.score))

    # 4) STY
    for r in rows: r.sty = sty_map.get(r.STY, 0)

    # 5) composite + fuzzy
    pool = top_by(rows, 'composite', fuzzy_pool)
    pool = fuzzy_rerank(query, keywords, pool)[:top_after_fuzzy]

    # 6) aggregate per code (vectorized via memmaps)
    per_code = aggregate(pool, q_vec, desc_vecs, kw_vecs, sty_map)
    ranked = sort(per_code, key='final')

    # 7) optional LLM final rerank
    if use_llm_rerank:
        ranked = LLM.rerank(query, entity_type, cands, ranked, k=topk)

    return ranked[:topk]
```

---

## 16) How This Meets the Hackathon Rubric

- **Accuracy (50%)**: Multi-signal scoring + STY coherence + fuzzy + optional LLM rerank.  
- **Technical Approach (25%)**: ANN indices, vectorized per‑code aggregation, concurrency controls, OOM‑aware embeddings.  
- **Innovation (15%)**: Weighted signal fusion, two‑stage fuzzy, stability boost on per‑code aggregation, strict XML I/O with LLM.  
- **Completeness & Docs (10%)**: Build/run scripts, README, this Architecture document, and column guide.

---

*End of document.*
