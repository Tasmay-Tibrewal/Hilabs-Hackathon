// ALL Deep Dive Content - Rich Visual Popups
import React from 'react'
import MermaidDiagram from '../components/MermaidDiagram'

export const deepDives: Record<string, { title: string, content: JSX.Element }> = {
  'string-fail': {
    title: 'Why String Matching Fails: Technical Analysis',
    content: (
      <div className="space-y-6">
        <div className="bg-red-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Levenshtein Distance Limitations</h4>
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg">
              <div className="font-bold text-xl mb-3">Example 1: Different Names</div>
              <div className="font-mono bg-gray-100 p-3 rounded mb-2">
                "paracetamol" vs "acetaminophen"
              </div>
              <div className="text-gray-700">
                Edit distance: 11 operations<br/>
                Similarity: ~15%<br/>
                <strong className="text-red-600">FAIL:</strong> Same drug, different names
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <div className="font-bold text-xl mb-3">Example 2: Abbreviations</div>
              <div className="font-mono bg-gray-100 p-3 rounded mb-2">
                "chest xr" vs "Plain X-ray of chest"
              </div>
              <div className="text-gray-700">
                Character overlap: minimal<br/>
                Similarity: ~25%<br/>
                <strong className="text-red-600">FAIL:</strong> Abbreviation not expanded
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">What's Missing?</h4>
          <ul className="space-y-3 text-xl text-gray-800">
            <li>✗ No semantic understanding of medical terms</li>
            <li>✗ Cannot handle synonyms or alternate names</li>
            <li>✗ No knowledge of brand names</li>
            <li>✗ Abbreviations completely break it</li>
            <li>✗ Word order matters too much</li>
            <li>✗ No domain-specific knowledge injection</li>
          </ul>
        </div>

        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Real Test Case</h4>
          <div className="bg-white p-6 rounded-lg">
            <div className="mb-4">
              <strong>Input:</strong> <code className="bg-gray-100 px-3 py-1 rounded">"ibuprofen (motrin) 20 mg/ml"</code>
            </div>
            <div className="mb-4">
              <strong>Target:</strong> <code className="bg-gray-100 px-3 py-1 rounded">"ibuprofen 20 MG/ML"</code>
            </div>
            <div className="p-4 bg-red-100 rounded">
              <strong>String similarity:</strong> ~60% (brand name "motrin" interferes)<br/>
              <strong>Our system:</strong> Correctly maps to RXNORM: 316073 ✓
            </div>
          </div>
        </div>
      </div>
    )
  },

  // Slide 4: Hybrid Methods — Dense Retrieval
  'method-faiss': {
    title: 'Dense Retrieval: FAISS + Embeddings',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Embedding Model</h4>
          <div className="grid grid-cols-2 gap-6 bg-white p-6 rounded-lg">
            <div>
              <div className="font-bold mb-2">Model</div>
              <div className="font-mono bg-gray-100 p-2 rounded">google/embeddinggemma-300m</div>
            </div>
            <div>
              <div className="font-bold mb-2">Dim / Norm</div>
              <div className="font-mono bg-gray-100 p-2 rounded">384 dims • L2-normalized</div>
            </div>
            <div className="col-span-2 text-gray-700">
              Open-source, compact, fast; good domain transfer for clinical strings
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">FAISS Index: HNSW (Flat-IP)</h4>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <div className="font-bold mb-2">Parameters</div>
              <div className="bg-gray-100 p-3 rounded font-mono text-sm">
                M=32 (neighbors/node)<br/>
                efConstruction=200 (build)<br/>
                efSearch=128 (query)
              </div>
            </div>
            <div>
              <div className="font-bold mb-2">Performance</div>
              <ul className="text-gray-700 space-y-1">
                <li>• O(log N) average search</li>
                <li>• Recall &gt; 95% with tuning</li>
                <li>• Cache-friendly graph hops</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-green-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Cosine via Inner Product</h4>
          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xl">
            cos(A,B) = A·B / (||A||×||B||)<br/>
            When ||A|| = ||B|| = 1 → cos(A,B) = A·B
          </div>
          <div className="mt-3 text-gray-700">We map to [0,1]: (cos+1)/2 and clip.</div>
        </div>
      </div>
    )
  },

  // Slide 4: Hybrid Methods — Fuzzy Matching
  'method-fuzzy': {
    title: 'Fuzzy Matching: Two-Stage Lexical',
    content: (
      <div className="space-y-6">
        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Why Two Stages?</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="space-y-2 text-lg text-gray-700">
              <li>• Stage 1: <strong>ratio</strong> for ultra-fast pruning</li>
              <li>• Stage 2: <strong>token_set_ratio</strong> for accuracy</li>
              <li>• Works well with abbreviations and reordered tokens</li>
            </ul>
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Flow Diagram</h4>
          <MermaidDiagram chart={`flowchart LR\n    Q["Anchors: query ∪ KWs"] --> P["Stage 1: ratio prefilter (top N per anchor)"]\n    P --> R["Reduced candidate set"]\n    R --> T["Stage 2: token_set_ratio (batched)"]\n    T --> M["Max over anchors -&gt; Fuzzy score"]\n    classDef a fill:#fef3c7,stroke:#f59e0b,stroke-width:3px\n    classDef b fill:#dbeafe,stroke:#3b82f6,stroke-width:3px\n    class Q a\n    class P,R,T,M b\n          `} />
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Anchors Strategy</h4>
          <div className="space-y-3">
            <div className="text-gray-700">Create multiple anchors: query ∪ LLM keywords</div>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-100 p-3 rounded font-mono text-sm">paracetamol 500</div>
              <div className="bg-blue-100 p-3 rounded font-mono text-sm">acetaminophen 500</div>
              <div className="bg-green-100 p-3 rounded font-mono text-sm">tylenol</div>
            </div>
            <div className="p-4 bg-purple-50 rounded">
              Final score = MAX similarity across all anchors
            </div>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Real Code (clean.py)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-x-auto">{`def fuzzy_scores_max(query: str, alt_keywords: List[str], choices: List[str],\n                     kw_limit: int = 8, workers: int = -1, prefilter: int = 200,\n                     use_two_stage: bool = True) -&gt; np.ndarray:\n    from rapidfuzz import process as rf_process, fuzz as rf_fuzz\n    anchors = [_rf_norm(query)]\n    for kw in alt_keywords:\n        t = _rf_norm(kw)\n        if len(t) >= 3 and t not in anchors:\n            anchors.append(t)\n        if len(anchors) >= 1 + kw_limit: break\n    choices_proc = [_rf_norm(c) for c in choices]\n    # Stage 1: ratio prefilter\n    keep = np.zeros(len(choices_proc), dtype=bool)\n    lim = min(prefilter, len(choices_proc))\n    for a in anchors:\n        arr = rf_process.cdist([a], choices_proc, scorer=rf_fuzz.ratio, workers=workers)[0]\n        idxs_top = np.argpartition(arr, -lim)[-lim:]\n        keep[idxs_top] = True\n    idxs = np.flatnonzero(keep); reduced = [choices_proc[i] for i in idxs]\n    # Stage 2: token_set_ratio on reduced\n    best = np.zeros(len(reduced), dtype=np.float32)\n    for a in anchors:\n        arr = rf_process.cdist([a], reduced, scorer=rf_fuzz.token_set_ratio, workers=workers)[0]\n        np.maximum(best, arr.astype(np.float32), out=best)\n    scores = np.zeros(len(choices_proc), dtype=np.float32)\n    scores[idxs] = best / 100.0\n    return scores`}</pre>
        </div>
      </div>
    )
  },

  // Slide 4: Hybrid Methods — LLM Intelligence
  'method-llm': {
    title: 'LLM Intelligence: Expansion and Rerank',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Two LLM Roles</h4>
          <div className="grid grid-cols-2 gap-6 bg-white p-6 rounded-lg">
            <div>
              <div className="font-bold text-purple-700 mb-2">1) Expansion</div>
              <div className="text-gray-700 text-lg">Generate synonyms, brands, abbreviations, and a short description + STY.</div>
            </div>
            <div>
              <div className="font-bold text-green-700 mb-2">2) Rerank (Optional)</div>
              <div className="text-gray-700 text-lg">Choose among 30 provided codes; no free-text -&gt; no hallucination.</div>
            </div>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Backends</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• vLLM (GPU) with Qwen3-4B</li>
            <li>• llama-cpp (CPU) with GGUF Qwen3-4B</li>
            <li>• Auto-selection by VRAM availability</li>
          </ul>
        </div>
      </div>
    )
  },

  // Slide 4: Hybrid Methods — Multi-Signal Scoring
  'method-scoring': {
    title: 'Multi-Signal Scoring: Weighted Blend',
    content: (
      <div className="space-y-6">
        <div className="bg-gradient-to-br from-blue-50 to-purple-50 p-6 rounded-xl border-4 border-purple-300">
          <h4 className="text-3xl font-bold mb-4">Composite Formula</h4>
          <div className="bg-white p-6 rounded-2xl font-mono text-2xl text-center">
            Score = 0.30×desc + 0.40×kw + 0.20×direct + 0.10×sty
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[{n:'Description',c:'blue',w:'30%'}, {n:'Keywords',c:'green',w:'40%'}, {n:'Direct',c:'yellow',w:'20%'}, {n:'STY',c:'purple',w:'10%'}].map(x => (
            <div key={x.n} className={`bg-white p-4 rounded-xl text-center border-4 border-${x.c}-200`}>
              <div className={`text-4xl font-bold text-${x.c}-600 mb-1`}>{x.w}</div>
              <div className="font-bold">{x.n}</div>
            </div>
          ))}
        </div>
      </div>
    )
  },

  // Slide 5: Build Phase — individual steps
  'build-load': {
    title: 'Build: Load Parquet Catalogs',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Inputs</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="space-y-2 text-lg text-gray-700">
              <li>• SNOMED (~900K rows)</li>
              <li>• RxNorm (~500K rows)</li>
              <li>• Parquet for fast IO</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Normalize</h4>
          <div className="font-mono bg-gray-100 p-3 rounded text-sm">[row_id, CODE, STR, CUI, TTY, STY, System]</div>
        </div>
      </div>
    )
  },
  'build-schema': {
    title: 'Build: Unified Schema + IDs',
    content: (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">row_id Mapping</h4>
          <div className="text-gray-700">Stable 0..N-1 IDs align FAISS vector ids with catalog rows.</div>
        </div>
        <div className="bg-green-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Why Uniform?</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Single pipeline for both systems</li>
            <li>• Simplifies aggregation and scoring</li>
          </ul>
        </div>
      </div>
    )
  },
  'build-embed': {
    title: 'Build: Embed Catalog Strings',
    content: (
      <div className="space-y-6">
        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Batching + OOM Backoff</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• Start with large batch; halve on CUDA OOM</li>
              <li>• Sync and clear cache between phases</li>
              <li>• Falls back to CPU if needed</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Normalization</h4>
          <div className="font-mono bg-gray-100 p-3 rounded text-sm">v = v / ||v||₂</div>
        </div>
      </div>
    )
  },
  'build-faiss': {
    title: 'Build: Create FAISS Index',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Index Types</h4>
          <div className="bg-white p-6 rounded-lg grid grid-cols-3 gap-4">
            <div>
              <div className="font-bold">HNSW (default)</div>
              <div className="text-gray-700 text-sm">Flat-IP, balanced recall/speed</div>
            </div>
            <div>
              <div className="font-bold">IVFPQ</div>
              <div className="text-gray-700 text-sm">Lower memory; needs training</div>
            </div>
            <div>
              <div className="font-bold">Flat-IP</div>
              <div className="text-gray-700 text-sm">Exact, high memory/latency</div>
            </div>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Persist</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Add vectors with ids (row_id)</li>
            <li>• Save `*.faiss` files per system</li>
            <li>• Record meta parameters</li>
          </ul>
        </div>
      </div>
    )
  },
  'build-memmap': {
    title: 'Build: Memory-Mapped Vectors (*.f32)',
    content: (
      <div className="space-y-6">
        <div className="bg-green-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Why Memmaps?</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• Exact vectors independent of FAISS type</li>
              <li>• Stream from disk; lower RAM</li>
              <li>• Enables vectorized per-code aggregation</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Sizes</h4>
          <div className="font-mono bg-gray-100 p-3 rounded text-sm">SNOMED ≈ 1.4GB • RxNorm ≈ 0.8GB</div>
        </div>
      </div>
    )
  },
  'build-sty': {
    title: 'Build: STY Vocabulary Embeddings',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Semantic Type (STY)</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• Embed unique STY strings present across catalogs</li>
              <li>• Store `sty_vocab.json` + `sty_embeddings.npy`</li>
              <li>• Used for fast type-compatibility scoring</li>
            </ul>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Alternatives: IVFPQ / Flat</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• IVFPQ: train on sample; lower memory; fast at large scale (tune nlist, pq_m, pq_nbits)</li>
            <li>• Flat-IP: exact search; highest memory and latency</li>
          </ul>
        </div>

        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Build Code (HNSW)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`import faiss
d = 384
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128
index.add_with_ids(vectors, row_ids)
faiss.write_index(index, 'snomed.index.faiss')`}</pre>
        </div>
      </div>
    )
  },

  // Slide 6: Pipeline overview
  'pipeline-detail': {
    title: 'Query Pipeline: Step-by-Step',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Flow</h4>
          <div className="bg-white p-6 rounded-lg grid grid-cols-2 gap-6">
            <div>
              <div className="font-bold mb-2">1) LLM Expansion</div>
              <div className="text-gray-700">keywords + description + STY hints</div>
            </div>
            <div>
              <div className="font-bold mb-2">2) Parallel FAISS</div>
              <div className="text-gray-700">4 signals × 2 systems -&gt; candidates</div>
            </div>
            <div>
              <div className="font-bold mb-2">3) Weighted Score</div>
              <div className="text-gray-700">normalize -&gt; blend to [0,1]</div>
            </div>
            <div>
              <div className="font-bold mb-2">4) Fuzzy Rerank</div>
              <div className="text-gray-700">anchors; MAX similarity</div>
            </div>
            <div>
              <div className="font-bold mb-2">5) Per-Code Aggregation</div>
              <div className="text-gray-700">stability over synonyms</div>
            </div>
            <div>
              <div className="font-bold mb-2">6) LLM Rerank (opt)</div>
              <div className="text-gray-700">choose among top 30 only</div>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Detailed Diagram</h4>
          <MermaidDiagram chart={`flowchart TD\n    Q["Query + Type"] --> X["LLM Expansion (XML): keywords, description, STY"]\n    X --> S1["Semantic Search (Direct)"]\n    X --> S2["Semantic Search (Description)"]\n    X --> S3["Semantic Search (Keywords, batched)"]\n    X --> S4["STY Compatibility"]\n    S1 --> M["Merge & Normalize -&gt; Composite Score"]\n    S2 --> M\n    S3 --> M\n    S4 --> M\n    M --> F["Fuzzy Two-Stage Re-rank"]\n    F --> A["Per-Code Aggregation"]\n    A --> R["Optional LLM Rerank (choose from list)"]\n    R --> K["Top-K Output"]\n    classDef a fill:#eef2ff,stroke:#6366f1,stroke-width:3px\n    classDef b fill:#dcfce7,stroke:#16a34a,stroke-width:3px\n    class Q,X a\n    class K b\n          `} />
        </div>
      </div>
    )
  },

  // Slide 6 (separate file): Semantic search signals
  'semantic-search': {
    title: 'Semantic Search: 4 Signals × 2 Systems',
    content: (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Signals</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Direct: embed raw query</li>
            <li>• Description: embed LLM summary</li>
            <li>• Keywords: batch embed all; take MAX</li>
            <li>• STY: compatibility via pre-embedded vocab</li>
          </ul>
        </div>
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Parallelization</h4>
          <div className="bg-white p-6 rounded-lg">Each signal searches SNOMED + RxNorm; results normalized and merged.</div>
        </div>
      </div>
    )
  },

  // Slide 7: LLM Query Expansion
  'llm-expansion': {
    title: 'LLM Query Expansion: XML Schema',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Correct XML Output Structure</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-auto">{`<candidate>
  <alternate_keywords>comma,separated,brands,common,scientific</alternate_keywords>
  <description>1-3 sentences describing the clinical meaning</description>
  <possible_semantic_term_types>choose ONLY from provided list</possible_semantic_term_types>
</candidate>
<candidate>...</candidate>`}</pre>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Why It Helps</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Bridges synonyms/brands/abbreviations</li>
            <li>• Supplies domain context via description</li>
            <li>• Provides type hints (STY) for scoring</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">System Prompt (Expansion)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`You are an expert clinical terminology normalizer.
Task: normalize messy clinical inputs into standardized concepts for mapping to RxNorm (medications) and SNOMED CT (labs, diagnoses, procedures).
Inputs may be abbreviated, colloquial, misspelled, or non-standard (e.g., "paracetamol" vs "Acetaminophen", "xr chest" vs "X-ray chest").
Return results STRICTLY as XML with one or more <candidate> blocks.
Each <candidate> block describes a distinct possible interpretation of the input.
It is optional to give multiple candidates, if you are sure about the meaning, a single candidate is fine. If unsure, 2-3 typical, up to 5.
For each candidate, include:
  <alternate_keywords>comma-separated short terms and brand/common names along with scientific names (give 2-5, and at max 10 alternate keywords)</alternate_keywords>
  <description>clear medical explanation (1-3 sentences)</description>
  <possible_semantic_term_types>comma-separated choices FROM THE PROVIDED LIST ONLY</possible_semantic_term_types>
Do not add any other tags. Do not include markdown. Use only ASCII punctuation.
Semantic type choices (union of RxNorm & SNOMED CT types): {sty_inventory}`}</pre>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">User Prompt (Expansion)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`Input Entity Description: {text}
Entity Type: {entity_type}

Output the XML now.`}</pre>
        </div>
      </div>
    )
  },

  // Slide 8: Scoring math
  'scoring-math': {
    title: 'Scoring Math: Normalization + Blend',
    content: (
      <div className="space-y-6">
        <div className="bg-gradient-to-br from-blue-50 to-purple-50 p-6 rounded-xl border-4 border-purple-300">
          <h4 className="text-3xl font-bold mb-4">Normalization</h4>
          <div className="bg-white p-6 rounded-lg font-mono text-sm">
            norm(x) = clip((x + 1) / 2, 0, 1) for cosine<br/>
            kw_score = MAX over keyword similarities
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Blend</h4>
          <div className="font-mono bg-gray-100 p-3 rounded text-xl text-center">0.30×desc + 0.40×kw + 0.20×direct + 0.10×sty</div>
          <div className="mt-3 text-gray-700">Weights chosen empirically; easy to tune.</div>
        </div>
      </div>
    )
  },

  // Slide 9: STY Compatibility
  'sty-deep': {
    title: 'STY Compatibility: Fast Type Safety',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Computation</h4>
          <div className="bg-white p-6 rounded-lg">
            <div className="font-mono text-xl mb-2">STY_score = MAX(cos(pred_STY, candidate_STY))</div>
            <div className="text-gray-700">Vectorized lookup via pre-embedded STY vocab; no re-embedding at query time.</div>
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Code Path (clean.py)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-x-auto">{`def precompute_sty_scores_map(pred_stys, sty_vocab, sty_emb) -&gt; Dict[str, float]:\n    # embed predicted STYs once; cosine vs all vocab; take max and map to [0,1]`}</pre>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Benefits</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Type-aware ranking</li>
            <li>• Soft matches (e.g., Clinical Drug ≈ Pharmacologic Substance)</li>
            <li>• Near-zero latency</li>
          </ul>
        </div>
      </div>
    )
  },

  // Slide 10: Fuzzy algorithm pseudocode
  'fuzzy-algo': {
    title: 'Two-Stage Fuzzy: Pseudocode',
    content: (
      <div className="space-y-6">
        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Anchored Rerank</h4>
          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm">
{`anchors = {query} ∪ llm_keywords\nfor a in anchors:\n  pool_a = top_k_by_ratio(catalog, a, k=200)\n  for cand in pool_a:\n    cand.score = max(cand.score, token_set_ratio(a, cand.STR))\nfinal_score = MAX over anchors`}
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Mermaid: Two-Stage</h4>
          <MermaidDiagram chart={`sequenceDiagram\n    participant U as User Query\n    participant A as Anchors\n    participant R as Ratio (Stage 1)\n    participant T as Token Set (Stage 2)\n    participant S as Score\n    U->>A: Build {query} ∪ KWs\n    A->>R: Parallel ratio cdist\n    R-->>A: Top-N per anchor\n    A->>T: token_set_ratio on reduced set\n    T-->>S: Take MAX across anchors\n          `} />
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Why It Works</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Token-set handles order/abbreviation</li>
            <li>• Using MAX across anchors is robust</li>
          </ul>
        </div>
      </div>
    )
  },


  // Slide 11: Per-code aggregation
  'aggregation': {
    title: 'Per-Code Aggregation: Stability Boost',
    content: (
      <div className="space-y-6">
        <div className="bg-orange-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Formula</h4>
          <div className="bg-white p-6 rounded-lg font-mono text-2xl text-center">
            final = avg_all × avg_top × √log₁₀(% in pool)
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Clean Implementation</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-x-auto">{`avg_all = mean(rows_all.composite)\navg_500 = mean(rows_top.composite)\npct = (len(rows_top)/len(rows_all))*100\nboost = sqrt(log10(max(1.0001, pct)))\nfinal = avg_all * avg_500 * boost`}</pre>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Diagram</h4>
          <MermaidDiagram chart={`flowchart LR\n    Synonyms["All rows for code"] --> A["avg_all"]\n    Pool["Top rows in fuzzy pool"] --> B["avg_top"]\n    B --> C["% in pool"]\n    C --> D["sqrt(log10(%))"]\n    A --> E["final score"]\n    D --> E\n          `} />
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Intuition</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Accounts for all synonyms (avg_all)</li>
            <li>• Rewards strong terms (avg_top)</li>
            <li>• Boosts codes well-represented in fuzzy pool</li>
          </ul>
        </div>
      </div>
    )
  },

  // Deep fuzzy dive: failures, pitfalls, parameters
  'fuzzy-deep': {
    title: 'Fuzzy Matching Deep Dive: Pitfalls, Params, Proof',
    content: (
      <div className="space-y-6">
        <div className="bg-red-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Common Failure Modes</h4>
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg">
              <div className="font-bold mb-2">Brand vs Generic</div>
              <div className="font-mono bg-gray-100 p-2 rounded text-sm">"motrin 200" vs "ibuprofen 200 MG Oral Tablet"</div>
              <div className="text-red-700 mt-2">Plain token overlap is weak without anchors.</div>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <div className="font-bold mb-2">Abbrev. Explosion</div>
              <div className="font-mono bg-gray-100 p-2 rounded text-sm">"ua w/ micro" vs "Urinalysis with reflex to microscopy"</div>
              <div className="text-red-700 mt-2">Set-based helps but needs seeded KWs.</div>
            </div>
          </div>
        </div>
        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Anchors = Query ∪ LLM Keywords</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• Limit keywords with <span className="font-mono">kw_limit</span> (default 8)</li>
              <li>• Drop anchors shorter than 3 chars</li>
              <li>• Normalize via RapidFuzz default_process</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Parameters (clean.py)</h4>
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded">
              <div className="font-bold">Two-Stage Prefilter</div>
              <div className="font-mono text-sm">prefilter = max(200, top_after * 2)</div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="font-bold">Workers</div>
              <div className="font-mono text-sm">workers = -1 (all cores)</div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="font-bold">Top After Fuzzy</div>
              <div className="font-mono text-sm">top_pool_after_fuzzy = 250</div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="font-bold">Fuzzy Pool</div>
              <div className="font-mono text-sm">fuzzy_pool = 500</div>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Proof-of-Benefit</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• Ratio stage avoids O(N×A) heavy set scoring</li>
              <li>• token_set_ratio robust to order and noise</li>
              <li>• MAX across anchors captures synonyms/brands</li>
            </ul>
          </div>
        </div>
      </div>
    )
  },

  // Deterministic mode (no LLM)
  'non-llm': {
    title: 'Deterministic Mode (No LLM): Fast, Stable, Cheap',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Pipeline</h4>
          <div className="bg-white p-6 rounded-lg">
            <ol className="list-decimal pl-6 space-y-2 text-lg text-gray-700">
              <li>Embed query; search SNOMED + RxNorm (FAISS)</li>
              <li>Optionally fuzzy rerank pooled rows</li>
              <li>Aggregate per-code and return Top-K</li>
            </ol>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">When to Use</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• CPU-only environments</li>
            <li>• Strict determinism requirements</li>
            <li>• Batch throughput without LLM cost</li>
          </ul>
        </div>
        <div className="bg-green-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Stability</h4>
          <div className="text-gray-700">Zero hallucinations; results depend solely on indices and fuzzy heuristics.</div>
        </div>
      </div>
    )
  },

  // Backends, quantization, and pooling
  'backends': {
    title: 'Inference Backends: vLLM, llama.cpp, Quantization',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Backends</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• vLLM (GPU): high throughput; bitsandbytes quant</li>
              <li>• llama-cpp (CPU): GGUF Qwen3-4B; runs anywhere</li>
              <li>• Auto-select by VRAM; override via flags</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Quantization</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• bitsandbytes (bnb) in vLLM for memory savings</li>
            <li>• GGUF quant levels in llama.cpp (e.g., Q4_K_XL)</li>
          </ul>
        </div>
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Concurrency & Pooling</h4>
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white p-4 rounded">
              <div className="font-bold mb-1">rows_concurrency</div>
              <div className="text-gray-700">Process multiple rows in parallel</div>
            </div>
            <div className="bg-white p-4 rounded">
              <div className="font-bold mb-1">llm_concurrency</div>
              <div className="text-gray-700">Bound LLM requests; avoid OOM</div>
            </div>
            <div className="bg-white p-4 rounded">
              <div className="font-bold mb-1">fuzzy_pool</div>
              <div className="text-gray-700">Rerank only top-N rows</div>
            </div>
            <div className="bg-white p-4 rounded">
              <div className="font-bold mb-1">top_pool_after_fuzzy</div>
              <div className="text-gray-700">Keep compact set for aggregation</div>
            </div>
          </div>
        </div>
      </div>
    )
  },

  // Slide 12: Additional architecture diagrams (safe labels)
  'arch-diagrams': {
    title: 'Architecture Deep Dive: Diagrams',
    content: (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Index Build (Offline)</h4>
          <MermaidDiagram chart={`flowchart LR\n  LoadParquet --> NormalizeSchema\n  NormalizeSchema --> EmbedSTR\n  EmbedSTR --> HNSWIndex\n  EmbedSTR --> VectorsF32\n  NormalizeSchema --> STYVocab\n  STYVocab --> EmbedSTY\n  HNSWIndex --> MetaJSON\n  VectorsF32 --> MetaJSON\n  EmbedSTY --> MetaJSON\n          `} />
          <div className="mt-4 text-gray-700 text-lg">
            Offline build produces three artifacts: FAISS indices (fast ANN), memmapped exact vectors (for vectorized math and aggregation), and pre-embedded STY vocabulary (for O(1) type compatibility).
            Schema normalization ensures consistent <span className="font-mono">row_id</span> across all artifacts for zero-copy joins.
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Query (Online)</h4>
          <MermaidDiagram chart={`flowchart TD\n  QueryInput --> ExpansionXML\n  ExpansionXML --> Direct\n  ExpansionXML --> Description\n  ExpansionXML --> Keywords\n  ExpansionXML --> STYMatch\n  Direct --> MergeScore\n  Description --> MergeScore\n  Keywords --> MergeScore\n  STYMatch --> MergeScore\n  MergeScore --> Fuzzy\n  Fuzzy --> AggregateCodes\n  AggregateCodes --> RerankLLM\n  RerankLLM --> TopK\n          `} />
          <div className="mt-4 text-gray-700 text-lg">
            Online path expands inputs (synonyms, brands, abbreviations, brief description, STY hints), performs four parallel semantic searches (direct, description, keywords) across both vocabularies, blends signals into a composite score, prunes and refines via two-stage fuzzy matching, aggregates per-code for stability, and optionally applies a constrained LLM reranker to select the final Top‑K.
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Concurrency Model</h4>
          <MermaidDiagram chart={`sequenceDiagram\n  participant UI as UI\n  participant JS as AsyncLoop\n  participant EMB as Embed\n  participant FAI as FAISS\n  participant FUZ as RapidFuzz\n  UI->>JS: Submit query\n  JS->>EMB: Semaphore(embed)\n  EMB-->>JS: vectors\n  JS->>FAI: Parallel searches\n  FAI-->>JS: candidates\n  JS->>FUZ: Two-stage rerank\n  FUZ-->>JS: fuzzy scores\n  JS->>UI: results\n          `} />
          <div className="mt-4 text-gray-700 text-lg">
            A two-level concurrency scheme: pipeline-level row parallelism and component-level semaphores. GPU-bound stages (embedding/LLM) sit behind semaphores; CPU-bound stages (FAISS, fuzzy) tune threads and use <span className="font-mono">to_thread</span> pools. This keeps VRAM stable while maximizing throughput.
          </div>
        </div>
      </div>
    )
  },

  // Slide 13: Concurrency
  'concurrency': {
    title: 'Concurrency: Pipelines and Semaphores',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Two Levels</h4>
          <div className="grid grid-cols-2 gap-6 bg-white p-6 rounded-lg">
            <div>
              <div className="font-bold text-blue-700">Pipeline</div>
              <div className="font-mono bg-gray-100 p-2 rounded text-sm">--rows_concurrency 128</div>
            </div>
            <div>
              <div className="font-bold text-green-700">Components</div>
              <div className="font-mono bg-gray-100 p-2 rounded text-sm">--llm_concurrency 200</div>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Semaphores (clean.py)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`_embed_sem = asyncio.Semaphore(emb_slots)
_faiss_sem = asyncio.Semaphore(faiss_slots)
async def embed_texts_async(...):
    async with _embed_sem:
        return await asyncio.to_thread(embed_texts, ...)
`}</pre>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">GPU/CPU Scheduling</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Embedding + LLM behind semaphores</li>
            <li>• FAISS threads tuned via OMP</li>
            <li>• ThreadPool for blocking tasks</li>
          </ul>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">CLI Arguments (clean.py)</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">Rows</div>
              <div className="font-mono">--rows_concurrency</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">LLM Concurrency</div>
              <div className="font-mono">--llm_concurrency</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">FAISS Threads</div>
              <div className="font-mono">--faiss_threads</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">BLAS Threads</div>
              <div className="font-mono">--blas_threads</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">CPU Pool</div>
              <div className="font-mono">--cpu_pool</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">Fuzzy Workers</div>
              <div className="font-mono">--fuzzy_workers</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">vLLM TP</div>
              <div className="font-mono">--llm_tp</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="font-bold mb-1">llama.cpp Threads</div>
              <div className="font-mono">--llm_n_threads</div>
            </div>
            <div className="bg-gray-50 p-3 rounded col-span-2">
              <div className="font-bold mb-1">vLLM Quantization</div>
              <div className="font-mono">--vllm_quantization</div>
            </div>
          </div>
        </div>
      </div>
    )
  },

  // Slide 14: Memory management
  'memory': {
    title: 'Memory Management: OOM Backoff + Memmaps',
    content: (
      <div className="space-y-6">
        <div className="bg-red-50 p-6 rounded-xl border-2 border-red-300">
          <h4 className="text-3xl font-bold mb-4 text-red-900">OOM Risks</h4>
          <ul className="text-lg text-gray-800 space-y-2">
            <li>• Embedding batch too large</li>
            <li>• LLM model VRAM heavy</li>
            <li>• Fragmentation over time</li>
          </ul>
        </div>
        <div className="bg-green-50 p-6 rounded-xl border-2 border-green-300">
          <h4 className="text-3xl font-bold mb-4 text-green-900">Mitigations</h4>
          <ul className="text-lg text-gray-800 space-y-2">
            <li>• Halving batch size on OOM</li>
            <li>• torch.cuda.empty_cache() / ipc_collect()</li>
            <li>• Memory-mapped vectors for aggregation</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Threading & BLAS (clean.py)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`def configure_cpu_threading(blas_threads=None, faiss_threads=None):
    if faiss_threads: faiss.omp_set_num_threads(int(faiss_threads))
    if blas_threads:
        for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(int(blas_threads))
`}</pre>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">CUDA Hygiene</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()`}</pre>
        </div>
      </div>
    )
  },

  // Slide 15: LLM rerank
  'llm-rerank': {
    title: 'LLM Rerank: Safe Code Selection',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Guardrails</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• Model sees <strong>up to 30 candidates</strong></li>
              <li>• Must pick from provided codes only</li>
              <li>• System preference by entity type</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Prompt Shape</h4>
          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-auto">
{`You are a clinical coding assistant.\nPick the BEST top-K codes that match the input.\nInput: "chest xr" (Procedure)\nCandidates:\n1) SNOMED 399208008 Plain X-ray of chest\n...\nAnswer ONLY with <choice> blocks.`}
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Rerank Diagram</h4>
          <MermaidDiagram chart={`flowchart TD\n    A["TopN per-code"] --> P["Format prompt with constraints"]\n    P --> L["LLM (pick top-k)"]\n    L --> V["Validate pickss"]\n    V --> O["Final ordered Top-K"]\n    classDef a fill:#fef3c7,stroke:#f59e0b,stroke-width:3px\n    class A,P,L,V,O a\n          `} />
        </div>
      </div>
    )
  },

  // Slide 17: Future work items
  'future-hnsw': {
    title: 'Enhanced HNSW: Small-World Topology',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Idea</h4>
          <div className="bg-white p-6 rounded-lg">Inject ~1% random long-range edges to reduce diameter; improves recall at fixed efSearch.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Plan</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Post-build augmentation pass</li>
            <li>• A/B recall vs latency evaluation</li>
            <li>• Tune percent of long-range edges (0.5–2%)</li>
            <li>• Guard with memory budget and build time caps</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Risks & Metrics</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Risk: over-densifying graph → memory bloat</li>
            <li>• Track: Recall@K vs efSearch; index size; build time</li>
          </ul>
        </div>
      </div>
    )
  },
  'future-rl': {
    title: 'LLM Fine-tuning: GRPO RL',
    content: (
      <div className="space-y-6">
        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Approach</h4>
          <div className="bg-white p-6 rounded-lg">Generate synthetic QA pairs; reinforce code-selection accuracy with GRPO; distill into 4B model.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Training Plan</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Generate synthetic pairs from gold mappings</li>
            <li>• Reward function: Top‑K inclusion + system preference</li>
            <li>• Distill into 4B GGUF for CPU fallback</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Metrics</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Top‑1/Top‑3 accuracy uplift vs baseline</li>
            <li>• Runtime cost (tokens/query) and throughput</li>
            <li>• Hallucination rate (expected 0 by constraints)</li>
          </ul>
        </div>
      </div>
    )
  },
  'future-desc': {
    title: 'Build-Time Descriptions per Code',
    content: (
      <div className="space-y-6">
        <div className="bg-green-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Richer Catalog</h4>
          <div className="bg-white p-6 rounded-lg">LLM-generate 1–2 sentence medical summaries for each code once; embed and persist to improve desc signal.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Scale</h4>
          <div className="text-gray-700">~60K calls for top-used codes; amortized offline cost. Deduplicate by CUI; batch with caching to minimize calls.</div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Design</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Store short summaries in a new column</li>
            <li>• Embed once; reuse in desc signal + STY hints</li>
            <li>• Flag provenance for auditing</li>
          </ul>
        </div>
      </div>
    )
  },
  'future-bm25': {
    title: 'BM25 Hybrid: Dense + Sparse',
    content: (
      <div className="space-y-6">
        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Motivation</h4>
          <div className="bg-white p-6 rounded-lg">Rare terms and exact numbers benefit from lexical BM25; combine with dense to boost recall.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Integration</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Maintain light BM25 index</li>
            <li>• Merge into candidate pool with weight</li>
            <li>• Downweight for generic/common terms</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Evaluation</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Lift on rare-token queries (numbers/units)</li>
            <li>• Latency overhead vs dense-only</li>
          </ul>
        </div>
      </div>
    )
  },
  'future-cluster': {
    title: 'Hierarchical Clustering by STY/Keywords',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Goal</h4>
          <div className="bg-white p-6 rounded-lg">Reduce search space by pre-grouping related codes; route queries to clusters.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Approach</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• STY → coarse buckets</li>
            <li>• KMeans/Annoy within buckets</li>
            <li>• Router picks top clusters per query</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Benefits</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Smaller candidate sets per query</li>
            <li>• Lower latency at similar recall</li>
          </ul>
        </div>
      </div>
    )
  },
  'future-cache': {
    title: 'Response Caching: 10× Speedup',
    content: (
      <div className="space-y-6">
        <div className="bg-green-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Cache Layers</h4>
          <div className="bg-white p-6 rounded-lg">
            <ul className="text-lg text-gray-700 space-y-2">
              <li>• LLM expansions by input hash</li>
              <li>• FAISS results by normalized query</li>
              <li>• Final top-k by query + entity type</li>
            </ul>
          </div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">System Prompt (Rerank)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`You are ranking standardized clinical codes for a messy input.
You receive the original input & entity type, plus up to 30 candidate codes (from RxNorm and SNOMED CT) with strings and types.
Goal: select the BEST top-N codes consistent with the input meaning and entity type.
Guidance:
- Prefer RxNorm for medications; prefer SNOMED CT for labs/diagnoses/procedures.
- However, if preferred coding system is missing or clearly worse, it's OK to choose the other system.
- Consider code string (STR), term type (TTY), semantic type (STY), and overall plausibility.
Return STRICT XML: a sequence of <choice> blocks; each contains:
  <code>THE EXACT CODE STRING</code>
  <reasoning>concise justification</reasoning>
Do not include other tags, no markdown, no extra text.`}</pre>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Cache Invalidation</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Version results by model/index build id</li>
            <li>• TTL for LLM expansions; LRU for Top‑K</li>
          </ul>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">User Prompt (Rerank)</h4>
          <pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xs overflow-x-auto">{`Original Query: {text}
Entity Type: {entity_type}

Expanded understanding (from earlier step):
{expanded_summary}

Candidate codes (system | code | STY | TTY | one or more names):
{candidates_blob}

Return the XML with up to {final_k} <choice> items.`}</pre>
        </div>
      </div>
    )
  },

  // Future: UMLS integration
  'future-umls': {
    title: 'UMLS Integration: Cross-Vocabulary Linking',
    content: (
      <div className="space-y-6">
        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Motivation</h4>
          <div className="bg-white p-6 rounded-lg">Leverage CUIs to unify SNOMED/RxNorm concepts and enable cross-terminology reasoning.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
          <h4 className="text-3xl font-bold mb-4">Plan</h4>
          <ul className="text-lg text-gray-700 space-y-2">
            <li>• Optional UMLS tables in build</li>
            <li>• CUI-aware scoring features</li>
            <li>• Safer system switching via CUI links</li>
          </ul>
        </div>
      </div>
    )
  },

  'embedding-fail': {
    title: 'Pure Embeddings: Technical Limitations',
    content: (
      <div className="space-y-6">
        <div className="bg-red-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">The Short Query Problem</h4>
          <div className="bg-white p-6 rounded-lg">
            <p className="text-xl mb-4">Transformers need context for good representations:</p>
            <div className="space-y-4">
              <div className="bg-green-100 p-4 rounded">
                <div className="font-mono text-lg mb-2">"Hemoglobin A1c measurement in blood"</div>
                <div className="text-green-700">✓ Good embedding (15+ tokens, rich context)</div>
              </div>
              <div className="bg-red-100 p-4 rounded">
                <div className="font-mono text-lg mb-2">"Hb"</div>
                <div className="text-red-700">✗ Poor embedding (2 chars, no context, ambiguous)</div>
              </div>
              <div className="bg-red-100 p-4 rounded">
                <div className="font-mono text-lg mb-2">"xr"</div>
                <div className="text-red-700">✗ Multiple meanings: x-ray? extended release? other?</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">No Medical Domain Knowledge</h4>
          <div className="bg-white p-6 rounded-lg">
            <p className="text-xl mb-4">Pre-trained models (even BERT-medical) don't know:</p>
            <ul className="space-y-3 text-gray-800 text-lg">
              <li>• "Paracetamol" = "Acetaminophen" (UK vs US terminology)</li>
              <li>• "Tylenol" = brand name for acetaminophen</li>
              <li>• "XR" commonly means "x-ray" in radiology context</li>
              <li>• "w/" is medical shorthand for "with"</li>
              <li>• "HbA1c" is hemoglobin A1c (glycated hemoglobin)</li>
              <li>• Standard dose formulations (500mg tablets, etc.)</li>
            </ul>
          </div>
        </div>

        <div className="bg-blue-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Embedding Model Used</h4>
          <div className="bg-white p-6 rounded-lg">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <div className="font-bold mb-2">Model:</div>
                <div className="font-mono bg-gray-100 p-2 rounded">google/embeddinggemma-300m</div>
              </div>
              <div>
                <div className="font-bold mb-2">Dimensions:</div>
                <div className="font-mono bg-gray-100 p-2 rounded">384</div>
              </div>
              <div>
                <div className="font-bold mb-2">Training:</div>
                <div className="text-gray-700">General domain + some medical</div>
              </div>
              <div>
                <div className="font-bold mb-2">Limitation:</div>
                <div className="text-red-600">Cannot capture ALL medical conventions</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  },

  'llm-fail': {
    title: 'Pure LLM Approach: Critical Failures',
    content: (
      <div className="space-y-6">
        <div className="bg-red-50 p-6 rounded-xl border-2 border-red-400">
          <h4 className="text-3xl font-bold mb-4 text-red-900">Hallucination Problem</h4>
          <div className="bg-white p-6 rounded-lg">
            <div className="font-bold text-xl mb-3">Real Scenario:</div>
            <div className="bg-gray-100 p-4 rounded-lg mb-4 font-mono text-sm">
              User: "Map 'chest xr' to SNOMED code"<br/>
              GPT-4: "The SNOMED code is 12345678 for Plain chest X-ray"
            </div>
            <div className="bg-red-100 p-4 rounded">
              <strong className="text-red-700">CRITICAL PROBLEM:</strong><br/>
              Code 12345678 doesn't exist in SNOMED CT!<br/>
              LLM invented a plausible-looking but completely fake code.
            </div>
            <div className="mt-4 text-gray-700">
              <strong>Why it happens:</strong> LLMs generate text based on patterns seen in training data, not retrieval from actual medical code databases.
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Context Window Limitation</h4>
          <div className="bg-white p-6 rounded-lg">
            <div className="text-xl mb-4">Cannot fit entire catalog in LLM context:</div>
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-100 p-4 rounded">
                <div className="font-bold mb-2">Our Catalog:</div>
                <div className="space-y-1 text-gray-700">
                  <div>• 1.4M rows total</div>
                  <div>• ~50 tokens/row average</div>
                  <div>• = <strong>70M tokens</strong></div>
                </div>
              </div>
              <div className="bg-gray-100 p-4 rounded">
                <div className="font-bold mb-2">GPT-4 Turbo:</div>
                <div className="space-y-1 text-gray-700">
                  <div>• 128K token context</div>
                  <div>• Can fit ~2,500 rows</div>
                  <div>• = <strong>0.18% of catalog!</strong></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 p-6 rounded-xl">
          <h4 className="text-3xl font-bold mb-4">Cost at Scale</h4>
          <div className="bg-white p-6 rounded-lg">
            <div className="text-xl mb-4">For 1,000 clinical queries:</div>
            <div className="space-y-3">
              <div className="flex justify-between items-center bg-red-100 p-4 rounded">
                <span className="font-bold">Pure GPT-4 API:</span>
                <span className="text-red-600 font-bold text-2xl">~$30-50</span>
              </div>
              <div className="flex justify-between items-center bg-green-100 p-4 rounded">
                <span className="font-bold">Our Hybrid (local LLM):</span>
                <span className="text-green-600 font-bold text-2xl">$0 (compute only)</span>
              </div>
            </div>
            <div className="mt-4 text-gray-600">
              At 100K queries/month: $3K-5K vs $0 -&gt; Massive savings!
            </div>
          </div>
        </div>
      </div>
    )
  }
}

// Export for use in presentation
export default deepDives
