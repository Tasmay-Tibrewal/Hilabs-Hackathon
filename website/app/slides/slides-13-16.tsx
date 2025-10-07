// Slides 13-16: Concurrency, Memory, LLM Rerank, Results

export function Slide13({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">Concurrency Architecture</h2>
      <p className="text-xl text-gray-700 mb-6">Async + semaphores balance GPU-heavy embedding/LLM work and CPU-heavy FAISS/fuzzy steps. BLAS and FAISS threads are capped for predictable throughput.</p>
      
      <div className="bg-blue-50 p-8 rounded-2xl border-4 border-blue-300 mb-8">
        <h3 className="text-3xl font-bold mb-6">Async/Await Pipeline</h3>
        <div className="grid grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl">
          <div className="text-xl font-bold mb-4">GPU Operations</div>
          <ul className="space-y-2 text-gray-700">
            <li>• Embedding: Semaphore(2×GPUs)</li>
            <li>• LLM: vLLM scheduler</li>
            <li>• Prevents OOM</li>
          </ul>
          <div className="text-sm text-gray-600 mt-2">LLM concurrency can be tuned via <span className="font-mono">--llm_concurrency</span>, and vLLM tensor parallel with <span className="font-mono">--llm_tp</span>.</div>
        </div>
        <div className="bg-white p-6 rounded-xl">
          <div className="text-xl font-bold mb-4">CPU Operations</div>
          <ul className="space-y-2 text-gray-700">
            <li>• FAISS: Tuned semaphore</li>
            <li>• Fuzzy: workers=-1</li>
            <li>• ThreadPool for blocking</li>
          </ul>
          <div className="text-sm text-gray-600 mt-2">Control FAISS threads with <span className="font-mono">--faiss_threads</span>, BLAS threads with <span className="font-mono">--blas_threads</span>, and CPU pool size with <span className="font-mono">--cpu_pool</span>.</div>
        </div>
      </div>
      </div>

      <div className="bg-green-50 p-8 rounded-2xl">
        <h3 className="text-3xl font-bold mb-6">Two-Level Concurrency</h3>
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl">
            <div className="font-bold text-green-700 text-xl mb-3">Pipeline Level</div>
            <div className="font-mono bg-gray-100 p-2 rounded mb-2">--rows_concurrency 128</div>
            <div className="text-gray-700">Process 128 rows simultaneously</div>
          </div>
          <div className="bg-white p-6 rounded-xl">
            <div className="font-bold text-blue-700 text-xl mb-3">Component Level</div>
            <div className="font-mono bg-gray-100 p-2 rounded mb-2">--llm_concurrency 200</div>
            <div className="text-gray-700">200 concurrent LLM requests</div>
          </div>
        </div>
      </div>

      <button onClick={() => onDetail('concurrency')} className="mt-6 w-full py-4 bg-blue-600 text-white text-xl font-bold rounded-xl hover:bg-blue-700">
        Implementation Details →
      </button>

      <div className="grid grid-cols-2 gap-4 mt-4">
        <button onClick={() => onDetail('backends')} className="w-full py-4 bg-purple-600 text-white text-xl font-bold rounded-xl hover:bg-purple-700">
          Backends & Quantization →
        </button>
        <button onClick={() => onDetail('memory')} className="w-full py-4 bg-red-600 text-white text-xl font-bold rounded-xl hover:bg-red-700">
          OOM & Memory Strategy →
        </button>
      </div>

      <div className="mt-6 bg-white p-6 rounded-2xl border-2 border-gray-200">
        <h3 className="text-2xl font-bold mb-4">Key CLI Arguments</h3>
        <div className="grid grid-cols-3 gap-4 text-gray-700 text-sm">
          <div><div className="font-bold">--rows_concurrency</div><div>Parallel rows</div></div>
          <div><div className="font-bold">--llm_concurrency</div><div>Concurrent LLM calls</div></div>
          <div><div className="font-bold">--faiss_threads</div><div>FAISS OMP threads</div></div>
          <div><div className="font-bold">--blas_threads</div><div>BLAS threads (OpenBLAS/MKL)</div></div>
          <div><div className="font-bold">--cpu_pool</div><div>to_thread pool size</div></div>
          <div><div className="font-bold">--fuzzy_workers</div><div>RapidFuzz workers</div></div>
          <div><div className="font-bold">--llm_tp</div><div>vLLM tensor parallel</div></div>
          <div><div className="font-bold">--llm_n_threads</div><div>llama-cpp threads</div></div>
          <div><div className="font-bold">--vllm_quantization</div><div>Quant mode (mxfp4/awq)</div></div>
        </div>
      </div>
    </div>
  )
}

export function Slide14({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-4">Memory Management</h2>
      <p className="text-xl text-gray-700 mb-6">CUDA OOM is mitigated with batch-size backoff, aggressive cache cleanup, and memmapped vectors. We also limit FAISS/BLAS threads and use quantized LLMs where appropriate.</p>
      
      <div className="bg-red-50 p-8 rounded-2xl border-4 border-red-300 mb-8">
        <h3 className="text-3xl font-bold mb-6 text-red-900">CUDA OOM Problem</h3>
        <ul className="space-y-3 text-xl text-gray-800">
          <li>• Embedding batch too large → crash</li>
          <li>• LLM model → 3-20GB VRAM</li>
          <li>• Fragmentation over time</li>
        </ul>
      </div>

      <div className="bg-green-50 p-8 rounded-2xl border-4 border-green-300">
        <h3 className="text-3xl font-bold mb-6 text-green-900">Our Solutions</h3>
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl">
            <div className="text-xl font-bold mb-4">Batch Backoff</div>
            <div className="text-gray-700">Halve batch size on OOM, retry down to bs=1</div>
          </div>
          <div className="bg-white p-6 rounded-xl">
            <div className="text-xl font-bold mb-4">Aggressive Cleanup</div>
            <div className="text-gray-700">torch.cuda.empty_cache() between phases</div>
          </div>
        </div>
      </div>

      <div className="bg-purple-50 p-8 rounded-2xl">
        <h3 className="text-3xl font-bold mb-6">Memory-Mapped Vectors</h3>
        <div className="bg-white p-6 rounded-xl">
          <div className="grid grid-cols-2 gap-6">
            <div className="font-mono text-sm space-y-1">
              <div>snomed_vectors.f32</div>
              <div>rxnorm_vectors.f32</div>
              <div>Shape: [n_rows, 384]</div>
            </div>
            <div className="text-gray-700">
              • Stream from disk<br/>
              • Exact embeddings<br/>
              • Fast vectorized ops
            </div>
          </div>
        </div>
      </div>

      {/* Detailed CLI arguments intentionally reside on Concurrency slide (Slide 13). */}

      <button onClick={() => onDetail('memory')} className="mt-6 w-full py-4 bg-red-600 text-white text-xl font-bold rounded-xl hover:bg-red-700">
        Full Strategy →
      </button>
    </div>
  )
}

export function Slide15({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Step 5: Optional LLM Rerank</h2>
      
      <div className="bg-purple-50 p-8 rounded-2xl border-4 border-purple-300 mb-8">
        <h3 className="text-3xl font-bold mb-6">Why Second LLM Call?</h3>
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl">
            <div className="text-xl font-bold mb-4 text-purple-700">First: Expansion</div>
            <div className="text-gray-700">Broadens search, adds knowledge</div>
          </div>
          <div className="bg-white p-6 rounded-xl">
            <div className="text-xl font-bold mb-4 text-green-700">Second: Rerank</div>
            <div className="text-gray-700">Narrows to best, applies preferences</div>
          </div>
        </div>
      </div>

      <div className="bg-white p-8 rounded-2xl border-4 border-gray-200">
        <h3 className="text-3xl font-bold mb-6">System Preference (Rules)</h3>
        <div className="bg-blue-50 p-6 rounded-xl text-lg">
          Default preference by entity type (RxNorm for meds; SNOMED CT otherwise) with smart switching:<br/>
          <span className="font-mono">switch</span> if alternative beats default by <span className="font-mono">--alt_margin</span> and meets <span className="font-mono">--min_score</span>.<br/>
          When LLM rerank is enabled, the prompt encodes these preferences so the model selects the preferred system unless evidence strongly favors the alternative.
        </div>
      </div>

      <div className="bg-black text-white p-8 rounded-2xl text-center">
        <div className="text-3xl font-bold">Anti-Hallucination</div>
        <div className="text-xl text-gray-300 mt-3">LLM can ONLY choose from 30 provided codes</div>
      </div>

      <button onClick={() => onDetail('llm-rerank')} className="mt-6 w-full py-4 bg-purple-600 text-white text-xl font-bold rounded-xl hover:bg-purple-700">
        Prompt Engineering →
      </button>
    </div>
  )
}

export function Slide16() {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Real Test Results</h2>
      
      <div className="grid grid-cols-2 gap-8">
        <div className="bg-green-50 p-8 rounded-2xl border-4 border-green-300">
          <h3 className="text-3xl font-bold mb-6 text-green-900">Strong Matches ✓</h3>
          <div className="space-y-3">
            {[
              { q: "coronary angioplasty", a: "Coronary angioplasty", c: "41339005" },
              { q: "allergic rhinitis", a: "Allergic rhinitis", c: "61582004" },
              { q: "acetaminophen 325 mg", a: "acetaminophen 325 MG Oral Tablet", c: "313782" },
            ].map((x, i) => (
              <div key={i} className="bg-white p-3 rounded-lg">
                <div className="font-mono text-sm mb-1">"{x.q}"</div>
                <div className="text-xs text-gray-600">→ {x.a} ({x.c})</div>
                <div className="text-green-600 font-bold text-sm mt-1">✓ Exact match</div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-yellow-50 p-8 rounded-2xl border-4 border-yellow-300">
          <h3 className="text-3xl font-bold mb-6 text-orange-900">Complex Cases ⚡</h3>
          <div className="space-y-3">
            {[
              { q: "urinalysis w/reflex microscopic", note: "Abbreviation expanded" },
              { q: "ibuprofen (motrin) 20 mg/ml", note: "Brand name handled" },
              { q: "partial thromboplastin time, activated", note: "Punctuation preserved" },
            ].map((x, i) => (
              <div key={i} className="bg-white p-3 rounded-lg">
                <div className="font-mono text-sm mb-1">"{x.q}"</div>
                <div className="text-orange-600 font-bold text-sm">⚡ {x.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-8 bg-black text-white p-8 rounded-2xl">
        <div className="grid grid-cols-4 gap-6 text-center">
          <div><div className="text-5xl font-bold">400</div><div className="text-gray-400">Test Cases</div></div>
          <div><div className="text-5xl font-bold">4</div><div className="text-gray-400">Entity Types</div></div>
          <div><div className="text-5xl font-bold">Both</div><div className="text-gray-400">Systems</div></div>
          <div><div className="text-5xl font-bold">Real</div><div className="text-gray-400">Data</div></div>
        </div>
      </div>
    </div>
  )
}
