// Slides 17-19: Future Work, Tech Stack, Summary

export function Slide17({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Future Enhancements</h2>
      
      <div className="grid grid-cols-2 gap-6">
        {[
          { icon: "🔬", title: "Enhanced HNSW", desc: "Inject ~1% long-range edges to shrink graph diameter and lift recall at fixed efSearch, with strict memory and build-time guards.", ref: "Watts–Strogatz small-world", id: "future-hnsw" },
          { icon: "🤖", title: "LLM Fine-tuning", desc: "GRPO on synthetic coding QA to reward correct Top‑K selection and system preference; distill to 4B GGUF for CPU fallback.", ref: "Domain adaptation", id: "future-rl" },
          { icon: "📝", title: "Rich Descriptions", desc: "Offline LLM summaries per code (1–2 sentences) embedded once to boost description+STY signals with provenance for audit.", ref: "~60K calls (top codes)", id: "future-desc" },
          { icon: "🔍", title: "BM25 Integration", desc: "Add lightweight BM25 for rare tokens/numbers and merge with dense candidates using learned weights.", ref: "Hybrid dense+sparse", id: "future-bm25" },
          { icon: "🌳", title: "Hierarchical Clustering", desc: "Route queries to STY-based buckets and sub-clusters (KMeans/Annoy) to cut candidate counts and latency.", ref: "Reduced search space", id: "future-cluster" },
          { icon: "💾", title: "Response Caching", desc: "Cache expansions, FAISS hits, and Top‑K by normalized key; version by model/index build id for safe invalidation.", ref: "Up to 10× speedup", id: "future-cache" },
          { icon: "🧠", title: "UMLS Integration", desc: "Use CUIs and MRSTY to unify synonyms across vocabularies, improve crosswalks, and reduce duplicates.", ref: "Cross‑vocab linking", id: "future-umls" },
        ].map((f, i) => (
          <div key={i} className="bg-white p-6 rounded-2xl border-2 border-gray-200 hover:border-black transition-all">
            <div className="flex items-start gap-4">
              <div className="text-4xl">{f.icon}</div>
              <div className="flex-1">
                <h3 className="text-2xl font-bold mb-2">{f.title}</h3>
                <p className="text-lg text-gray-700 mb-2">{f.desc}</p>
                <div className="text-sm text-gray-500">{f.ref}</div>
              </div>
            </div>
            <button onClick={() => onDetail(f.id)} className="mt-4 px-4 py-2 bg-gray-100 rounded-lg hover:bg-gray-200 font-bold text-sm w-full">
              Technical Details →
            </button>
          </div>
        ))}
      </div>

      <div className="mt-8 bg-black text-white p-6 rounded-xl text-center">
        <div className="text-3xl font-bold">Production-Ready Today</div>
        <div className="text-xl text-gray-300 mt-2">Enhancements are incremental</div>
      </div>
    </div>
  )
}

export function Slide18() {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Technology Stack</h2>
      
      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="bg-blue-50 p-6 rounded-2xl border-2 border-blue-300">
          <h3 className="text-2xl font-bold mb-4">Core Libraries</h3>
          <ul className="space-y-2 text-lg">
            <li>• FAISS (Meta)</li>
            <li>• SentenceTransformers</li>
            <li>• RapidFuzz</li>
            <li>• PyTorch</li>
            <li>• NumPy/Pandas</li>
          </ul>
        </div>

        <div className="bg-green-50 p-6 rounded-2xl border-2 border-green-300">
          <h3 className="text-2xl font-bold mb-4">LLM Backends</h3>
          <ul className="space-y-2 text-lg">
            <li>• vLLM (GPU)</li>
            <li>• llama-cpp (CPU)</li>
            <li>• Qwen3 models</li>
            <li>• HuggingFace Hub</li>
          </ul>
        </div>

        <div className="bg-purple-50 p-6 rounded-2xl border-2 border-purple-300">
          <h3 className="text-2xl font-bold mb-4">Vocabularies</h3>
          <ul className="space-y-2 text-lg">
            <li>• RxNorm (NLM)</li>
            <li>• SNOMED CT</li>
          </ul>
        </div>
      </div>

      <div className="bg-black text-white p-8 rounded-2xl mb-8">
        <h3 className="text-3xl font-bold mb-6 text-center">100% Open Source</h3>
        <div className="grid grid-cols-2 gap-6 text-xl text-center">
          <div>✓ No proprietary APIs</div>
          <div>✓ All processing local</div>
        </div>
      </div>

      <div className="bg-yellow-50 p-6 rounded-2xl border-2 border-yellow-300">
        <h3 className="text-2xl font-bold mb-4">Compliance</h3>
        <ul className="space-y-2 text-lg text-gray-700">
          <li>• <strong>SNOMED CT:</strong> Licensed - verify usage rights</li>
          <li>• <strong>RxNorm:</strong> U.S. NLM - check terms</li>
          <li>• <strong>Privacy:</strong> No PHI persisted</li>
        </ul>
      </div>
    </div>
  )
}

export function Slide19() {
  return (
    <div className="max-w-6xl text-center">
      <h2 className="text-6xl font-bold mb-12">Summary</h2>
      <div className="text-3xl text-gray-700 mb-8 leading-relaxed">
        Production-ready <strong>hybrid system</strong> combining<br/>
        dense retrieval, fuzzy matching, LLM intelligence, and multi-signal scoring
      </div>
      <div className="bg-black text-white p-12 rounded-3xl mb-8">
        <div className="grid grid-cols-3 gap-8">
          <div>
            <div className="text-5xl font-bold mb-3">100%</div>
            <div className="text-gray-400 text-xl">Open Source</div>
          </div>
          <div>
            <div className="text-5xl font-bold mb-3">1.4M</div>
            <div className="text-gray-400 text-xl">Entries Indexed</div>
          </div>
          <div>
            <div className="text-5xl font-bold mb-3">Hybrid</div>
            <div className="text-gray-400 text-xl">4 Methods</div>
          </div>
        </div>
      </div>
      <div className="text-4xl font-bold mb-4">Production-Ready Today</div>
      <div className="text-3xl text-gray-600">Extensible for Tomorrow</div>
      <div className="text-2xl text-gray-500 mt-8">Thank you!</div>
    </div>
  )
}
