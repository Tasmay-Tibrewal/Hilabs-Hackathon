// Slides 5-8: Build Phase, LLM Expansion, Semantic Search, Scoring

import MermaidDiagram from '../components/MermaidDiagram'

export function Slide5({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Build Phase: Index Creation</h2>
      <p className="text-2xl text-gray-600 mb-8">One-time offline processing that makes query-time blazing fast: clean schema, embed strings, build FAISS, persist exact vectors, and precompute STY embeddings.</p>
      
      <div className="space-y-4">
        {[
          { num: 1, title: "Load Parquet Catalogs", desc: "SNOMED (~900K) + RxNorm (~500K)", hint: "Fast IO; preserve core columns", id: "build-load" },
          { num: 2, title: "Normalize Schema", desc: "[row_id, CODE, STR, CUI, TTY, STY, System]", hint: "Stable row_id aligns FAISS IDs", id: "build-schema" },
          { num: 3, title: "Embed All Strings", desc: "SentenceTransformer with OOM backoff", hint: "L2-normalize vectors; batch halving on OOM", id: "build-embed" },
          { num: 4, title: "Build FAISS Index", desc: "HNSW (M=32, efConstruction=200)", hint: "efSearch tunes recall/latency", id: "build-faiss" },
          { num: 5, title: "Memory-Mapped Vectors", desc: "*.f32 files for fast aggregation", hint: "Exact vectors for vectorized scoring", id: "build-memmap" },
          { num: 6, title: "STY Vocabulary", desc: "Embed ~200-300 semantic types", hint: "Precompute for O(1) lookups", id: "build-sty" },
        ].map(s => (
          <div key={s.num} className="bg-white border-2 border-gray-200 p-6 rounded-xl flex items-center justify-between hover:border-black transition-all">
            <div className="flex items-center gap-6 flex-1">
              <div className="text-4xl font-bold text-blue-600 w-16">{s.num}</div>
              <div>
                <div className="text-2xl font-bold mb-1">{s.title}</div>
                <div className="text-lg text-gray-600">{s.desc}</div>
                <div className="text-sm text-gray-500">{s.hint}</div>
              </div>
            </div>
            <button onClick={() => onDetail(s.id)} className="px-4 py-2 bg-gray-100 rounded-lg hover:bg-gray-200 font-bold">
              Details →
            </button>
          </div>
        ))}
      </div>

      <div className="mt-8 bg-gray-900 text-white p-6 rounded-xl">
        <div className="grid grid-cols-3 gap-6 text-center">
          <div><div className="text-3xl font-bold">~10-60 min</div><div className="text-gray-400">Build Time</div></div>
          <div><div className="text-3xl font-bold">~10 GB</div><div className="text-gray-400">Artifacts</div></div>
          <div><div className="text-3xl font-bold">One-Time</div><div className="text-gray-400">Frequency</div></div>
        </div>
      </div>

      <div className="mt-8 bg-white p-6 rounded-xl border-2 border-gray-200">
        <h3 className="text-2xl font-bold mb-4">Fuzzy Matching</h3>
        <ul className="text-lg text-gray-700 space-y-2">
          <li>• Two-stage RapidFuzz: ratio prefilter then token_set_ratio on reduced candidates.</li>
          <li>• Anchors = {`{query} ∪ keywords`} with MAX across anchors to capture brands/abbreviations.</li>
          <li>• Highly parallel via workers with vectorized scoring on reduced pool.</li>
        </ul>
        <button onClick={() => onDetail('fuzzy-deep')} className="mt-4 w-full py-3 bg-yellow-600 text-white rounded-lg font-bold hover:bg-yellow-700">Fuzzy Matching Deep Dive →</button>
      </div>
    </div>
  )
}

export function Slide6({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-12">Query Pipeline Overview</h2>
      
      <div className="bg-gray-50 p-8 rounded-2xl mb-8">
        <MermaidDiagram chart={`flowchart TD
    A["Raw Query + Entity Type"]
    A --> B["LLM Expansion"]
    B --> C["4 Parallel FAISS Searches"]
    C --> D["Weighted Scoring"]
    D --> E["Fuzzy Re-rank"]
    E --> F["Per-Code Aggregation"]
    F --> G["Optional LLM Rerank"]
    G --> H["Top-K Output"]
    
    style A fill:#fef3c7,stroke:#f59e0b,stroke-width:4px
    style H fill:#d1fae5,stroke:#10b981,stroke-width:4px
    style B fill:#dbeafe,stroke:#3b82f6,stroke-width:3px
    style G fill:#e9d5ff,stroke:#a855f7,stroke-width:3px`} />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-blue-50 p-6 rounded-xl border-2 border-blue-200">
          <h3 className="text-2xl font-bold mb-4">Innovation #1</h3>
          <p className="text-xl text-gray-700 mb-3"><strong>Search BOTH systems</strong></p>
          <p className="text-gray-600">Let scoring decide best match, not entity type alone</p>
        </div>
        <div className="bg-green-50 p-6 rounded-xl border-2 border-green-200">
          <h3 className="text-2xl font-bold mb-4">Innovation #2</h3>
          <p className="text-xl text-gray-700 mb-3"><strong>Per-code aggregation</strong></p>
          <p className="text-gray-600">Aggregate all synonym rows for stability</p>
        </div>
      </div>

      <button onClick={() => onDetail('pipeline-detail')} className="mt-6 w-full py-4 bg-black text-white text-xl font-bold rounded-xl hover:bg-gray-800">
        Step-by-Step Breakdown →
      </button>
    </div>
  )
}

export function Slide7({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">Deterministic (No LLM) Mode</h2>
      <p className="text-xl text-gray-700 mb-8">Use this mode for CPU-only or cost-sensitive runs. It performs semantic retrieval on both vocabularies, optional fuzzy re-rank, and per-code aggregation — completely deterministic, zero hallucinations.</p>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-xl border-2 border-gray-200">
          <div className="text-2xl font-bold mb-2">1) Direct Semantic</div>
          <div className="text-gray-700">Embed the query and search RxNorm + SNOMED. Normalize cosine to [0,1].</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-2 border-gray-200">
          <div className="text-2xl font-bold mb-2">2) Fuzzy Re-rank</div>
          <div className="text-gray-700">Optional RapidFuzz two-stage rerank to refine top pool efficiently.</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-2 border-gray-200">
          <div className="text-2xl font-bold mb-2">3) Aggregate Per-Code</div>
          <div className="text-gray-700">Stabilize with synonym pooling: avg_all × avg_top × √log₁₀(% in pool).</div>
        </div>
      </div>

      <div className="bg-gray-900 text-white p-8 rounded-2xl">
        <h3 className="text-3xl font-bold mb-4">When to Use</h3>
        <ul className="space-y-2 text-lg">
          <li>• Strict determinism (audits, baselines)</li>
          <li>• CPU-only environments (llama.cpp disabled)</li>
          <li>• Batch throughput with predictable cost</li>
        </ul>
      </div>

      <div className="mt-8 bg-white p-6 rounded-xl border-2 border-gray-200">
        <h3 className="text-2xl font-bold mb-4">Four Methods Summary</h3>
        <ul className="text-lg text-gray-700 space-y-2">
          <li>• Dense Retrieval: semantic nearest neighbors over RxNorm/SNOMED using a compact 384‑d embedding model.</li>
          <li>• Fuzzy Matching: two-stage RapidFuzz (ratio prefilter → token_set_ratio refine) to re‑rank a large pool.</li>
          <li>• LLM Intelligence: expand short/ambiguous inputs into keywords + description + STY; optional final rerank.</li>
          <li>• Multi-Signal Scoring: blend description, keywords, direct, and STY into one robust composite score.</li>
        </ul>
      </div>

      <button onClick={() => onDetail('non-llm')} className="mt-6 w-full py-4 bg-gray-900 text-white text-xl font-bold rounded-xl hover:bg-gray-800">
        Deterministic Pipeline Details →
      </button>
    </div>
  )
}

export function Slide8({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">Step 2: Multi-Signal Scoring</h2>
      <p className="text-xl text-gray-700 mb-6">We blend four signals — description, keywords, direct query, and STY compatibility — into one robust score. Any single strong signal can carry ambiguous inputs, while the others corroborate and improve stability.</p>
      
      <div className="bg-gradient-to-br from-blue-50 to-purple-50 p-12 rounded-3xl border-4 border-purple-300 mb-8">
        <div className="text-center mb-8">
          <div className="text-4xl font-bold mb-6">Composite Formula</div>
          <div className="bg-white p-8 rounded-2xl font-mono text-3xl border-4 border-gray-300">
            Score = 0.30×desc + 0.40×kw + 0.20×direct + 0.10×sty
          </div>
        </div>

        <div className="grid grid-cols-4 gap-6">
          {[
            { w: "30%", name: "Description", color: "blue" },
            { w: "40%", name: "Keywords", color: "green" },
            { w: "20%", name: "Direct", color: "yellow" },
            { w: "10%", name: "STY", color: "purple" },
          ].map(s => (
            <div key={s.name} className={`bg-white p-6 rounded-xl text-center border-4 border-${s.color}-200`}>
              <div className={`text-5xl font-bold text-${s.color}-600 mb-2`}>{s.w}</div>
              <div className="font-bold text-xl">{s.name}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-900 text-white p-8 rounded-2xl">
        <h3 className="text-3xl font-bold mb-6">Parallel FAISS Searches</h3>
        <div className="grid grid-cols-2 gap-6 text-lg">
          <div>
            <div className="font-bold mb-3">For Each Signal:</div>
            <div className="text-gray-300">• Search SNOMED (top-500)<br/>• Search RxNorm (top-500)<br/>• Normalize to [0,1]</div>
          </div>
          <div>
            <div className="font-bold mb-3">Result:</div>
            <div className="text-gray-300">• Up to 4,000 candidates (4 signals × 2 systems × 500)<br/>• Deduplicated<br/>• Weighted combination</div>
          </div>
        </div>
      </div>

      <button onClick={() => onDetail('scoring-math')} className="mt-6 w-full py-4 bg-orange-600 text-white text-xl font-bold rounded-xl hover:bg-orange-700">
        Mathematical Details →
      </button>
    </div>
  )
}
