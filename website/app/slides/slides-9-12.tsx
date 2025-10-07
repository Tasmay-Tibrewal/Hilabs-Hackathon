// Slides 9-12: STY Compatibility, Fuzzy Matching, Aggregation, Architecture

import MermaidDiagram from '../components/MermaidDiagram'

export function Slide9({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">STY Compatibility Innovation</h2>
      <p className="text-xl text-gray-700 mb-6">Semantic Type (STY) provides a type-safety prior. We predict plausible STYs from expansion and match against candidate STYs quickly using pre-embedded vocab (no per-row embedding at query time). With 4 signals × 2 systems × top-500 each, we have ~4,000 base candidates; allowing up to 5 candidates per query yields up to <strong>~20,000</strong> comparisons handled efficiently.</p>
      
      <div className="bg-purple-50 p-8 rounded-2xl border-4 border-purple-300 mb-8">
        <h3 className="text-3xl font-bold mb-6">What is STY?</h3>
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl">
            <div className="text-xl font-bold mb-4">Examples:</div>
            <div className="space-y-2">
              <div className="px-3 py-2 bg-blue-100 rounded">Pharmacologic Substance</div>
              <div className="px-3 py-2 bg-green-100 rounded">Disease or Syndrome</div>
              <div className="px-3 py-2 bg-purple-100 rounded">Diagnostic Procedure</div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-xl">
            <div className="text-xl font-bold mb-4">Purpose:</div>
            <ul className="space-y-2 text-gray-700">
              <li>• Clinical category</li>
              <li>• Type safety</li>
              <li>• Soft matching</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white p-8 rounded-2xl border-4 border-gray-200">
        <h3 className="text-3xl font-bold mb-6">Computation Method</h3>
        <div className="bg-purple-50 p-6 rounded-xl">
          <div className="font-mono text-xl mb-3">STY_score = MAX(cos(pred_STY, candidate_STY))</div>
          <div className="text-gray-700">
            <strong>Fast:</strong> Vectorized via pre-embedded vocab<br/>
            <strong>Soft match:</strong> "Clinical Drug" ≈ 0.85 vs "Pharmacologic Substance"
          </div>
        </div>
      </div>

      <button onClick={() => onDetail('sty-deep')} className="mt-6 w-full py-4 bg-purple-600 text-white text-xl font-bold rounded-xl hover:bg-purple-700">
        Implementation Code →
      </button>
    </div>
  )
}

export function Slide10({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">Step 3: Two-Stage Fuzzy</h2>
      <p className="text-xl text-gray-700 mb-6">We first prune candidates with a very fast ratio scorer, then refine with token_set_ratio on the reduced set. Anchors = {`{query} ∪ keywords`} ensure brands and abbreviations are covered, and we take MAX over anchors. With up to 4k base candidates, and up to 5 LLM candidates per input (when enabled), the fuzzy pass may handle up to ~20k string comparisons efficiently.</p>
      
      <div className="bg-white p-8 rounded-2xl border-4 border-gray-200 mb-8">
        <h3 className="text-3xl font-bold mb-6 text-center">Two-Stage Process</h3>
        <div className="grid grid-cols-2 gap-8">
          <div className="bg-orange-50 p-6 rounded-xl border-2 border-orange-300">
            <div className="text-2xl font-bold mb-4 text-orange-700">Stage 1: Ratio</div>
            <div className="space-y-2">
              <div><strong>Speed:</strong> ⚡ Very fast</div>
              <div><strong>Method:</strong> Edit distance</div>
              <div><strong>Action:</strong> Keep top-200/anchor</div>
            </div>
          </div>

          <div className="bg-green-50 p-6 rounded-xl border-2 border-green-300">
            <div className="text-2xl font-bold mb-4 text-green-700">Stage 2: Token Set</div>
            <div className="space-y-2">
              <div><strong>Accuracy:</strong> 🎯 Precise</div>
              <div><strong>Method:</strong> Set-based</div>
              <div><strong>Final:</strong> MAX across anchors</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 p-8 rounded-2xl">
        <h3 className="text-3xl font-bold mb-6">Anchors Explained</h3>
        <p className="text-gray-700 mb-4">Anchors are alternate strings (query plus keywords). Each anchor votes; final fuzzy score is the maximum across anchors.</p>
        <div className="bg-white p-6 rounded-xl">
          <div className="space-y-2 text-lg">
            <div className="px-4 py-2 bg-gray-100 rounded font-mono">paracetamol 500</div>
            <div className="px-4 py-2 bg-blue-100 rounded font-mono">acetaminophen 500</div>
            <div className="px-4 py-2 bg-green-100 rounded font-mono">tylenol</div>
          </div>
          <div className="mt-4 p-4 bg-yellow-50 rounded">
            <strong>Final score</strong> = MAX similarity across all anchors
          </div>
        </div>
      </div>

      <button onClick={() => onDetail('fuzzy-algo')} className="mt-6 w-full py-4 bg-yellow-600 text-white text-xl font-bold rounded-xl hover:bg-yellow-700">
        Algorithm Pseudocode →
      </button>
    </div>
  )
}

export function Slide11({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">Step 4: Per-Code Aggregation</h2>
      <p className="text-xl text-gray-700 mb-6">We aggregate evidence across all synonyms of the same code to produce a stable final score, then boost codes that are consistently present in the fuzzy pool.</p>
      
      <div className="bg-orange-50 p-8 rounded-2xl border-4 border-orange-300 mb-8">
        <h3 className="text-3xl font-bold mb-6">Why Aggregate?</h3>
        <div className="bg-white p-6 rounded-xl">
          <div className="text-xl mb-4">Multiple rows per code (synonyms):</div>
          <div className="bg-gray-50 p-4 rounded font-mono text-sm space-y-1">
            <div>Code 198440: "Acetaminophen 500 MG Oral Tablet"</div>
            <div>Code 198440: "APAP 500 MG tablet"</div>
            <div>Code 198440: "Paracetamol 500mg oral"</div>
          </div>
          <div className="mt-4"><strong>Solution:</strong> Aggregate all rows!</div>
        </div>
      </div>

      <div className="bg-white p-8 rounded-2xl border-4 border-gray-200">
        <h3 className="text-3xl font-bold mb-6 text-center">Aggregation Formula</h3>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-8 rounded-2xl mb-6">
          <div className="text-center font-mono text-3xl mb-6 font-bold">
            final = avg_all × avg_top × √log₁₀(% in pool)
          </div>
          <div className="grid grid-cols-3 gap-6">
            <div className="bg-white p-4 rounded-xl text-center">
              <div className="font-bold text-blue-600">avg_all</div>
              <div className="text-sm text-gray-600">Mean over all rows</div>
            </div>
            <div className="bg-white p-4 rounded-xl text-center">
              <div className="font-bold text-green-600">avg_top</div>
              <div className="text-sm text-gray-600">Mean in fuzzy pool</div>
            </div>
            <div className="bg-white p-4 rounded-xl text-center">
              <div className="font-bold text-purple-600">boost</div>
              <div className="text-sm text-gray-600">Stability factor</div>
            </div>
          </div>
        </div>
      </div>

      <button onClick={() => onDetail('aggregation')} className="mt-6 w-full py-4 bg-orange-600 text-white text-xl font-bold rounded-xl hover:bg-orange-700">
        Mathematical Derivation →
      </button>
    </div>
  )
}

export function Slide12({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-7xl w-full">
      <h2 className="text-6xl font-bold mb-8 text-center">Complete System Architecture</h2>
      
      <div className="bg-gray-50 p-8 rounded-3xl border-4 border-gray-300">
        <MermaidDiagram chart={`flowchart TD
    A["Input Query"]
    A --> B["LLM Expansion"]
    B --> C1["FAISS SNOMED"]
    B --> C2["FAISS RxNorm"]
    C1 --> D["Weighted Scoring"]
    C2 --> D
    D --> E["Fuzzy Pool 500"]
    E --> F["Per-Code Aggregation"]
    F --> G["Top-K Output"]
    
    style A fill:#fef3c7,stroke:#f59e0b,stroke-width:4px
    style G fill:#d1fae5,stroke:#10b981,stroke-width:4px
    style B fill:#dbeafe,stroke:#3b82f6,stroke-width:3px`} />
      </div>
      <p className="text-xl text-gray-700 mt-6">The pipeline expands inputs, runs parallel semantic searches on both vocabularies, blends multi-signal scores, prunes and refines via fuzzy matching, aggregates by code for stability, and optionally re-ranks the final Top‑K with a constrained LLM.</p>

      <div className="mt-8 grid grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl border-2 border-blue-200 text-center">
          <div className="text-3xl font-bold text-blue-600">Parallel</div>
          <div className="text-gray-700">Async FAISS lookups across signals and systems</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-2 border-green-200 text-center">
          <div className="text-3xl font-bold text-green-600">Efficient</div>
          <div className="text-gray-700">Memmapped vectors enable fast recomputation</div>
        </div>
        <div className="bg-white p-6 rounded-xl border-2 border-purple-200 text-center">
          <div className="text-3xl font-bold text-purple-600">Scalable</div>
          <div className="text-gray-700">HNSW achieves ~O(log N) average search</div>
        </div>
      </div>

      <button onClick={() => onDetail('arch-diagrams')} className="mt-8 w-full py-4 bg-gray-900 text-white text-xl font-bold rounded-xl hover:bg-gray-800">
        More Architecture Diagrams →
      </button>
    </div>
  )
}
