// Slide 6: Semantic Search Details

export function Slide6({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Semantic Search: 4 Parallel Signals</h2>
      
      <div className="bg-blue-50 p-8 rounded-2xl border-4 border-blue-300 mb-8">
        <h3 className="text-3xl font-bold mb-6">How We Search</h3>
      <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl">
            <div className="text-2xl font-bold mb-4 text-blue-600">1. Direct Query</div>
            <div className="text-gray-700 mb-3">
              Embed the raw input: "chest xr"<br/>
              Search both SNOMED + RxNorm indices<br/>
              Get top-500 similar terms
            </div>
            <div className="text-sm text-gray-600">Intuition: Retains exact user wording; strong when query is descriptive.</div>
            <div className="bg-gray-100 p-3 rounded font-mono text-sm">
              direct_score = cos(query_vec, catalog_vec)
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl">
            <div className="text-2xl font-bold mb-4 text-green-600">2. Description Match</div>
            <div className="text-gray-700 mb-3">
              Embed LLM description<br/>
              Search both indices<br/>
              Captures medical semantics
            </div>
            <div className="text-sm text-gray-600">Intuition: Supplies domain context; robust for short/ambiguous inputs.</div>
            <div className="bg-gray-100 p-3 rounded font-mono text-sm">
              desc_score = cos(desc_vec, catalog_vec)
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl">
            <div className="text-2xl font-bold mb-4 text-purple-600">3. Keyword Batch</div>
            <div className="text-gray-700 mb-3">
              Embed ALL keywords together<br/>
              Batch search (faster!)<br/>
              Take MAX over all keywords
            </div>
            <div className="text-sm text-gray-600">Intuition: Covers synonyms, brands, abbreviations; one hit is enough.</div>
            <div className="bg-gray-100 p-3 rounded font-mono text-sm">
              kw_score = MAX(cos(kw_vecs, catalog_vec))
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl">
            <div className="text-2xl font-bold mb-4 text-orange-600">4. STY Compatibility</div>
            <div className="text-gray-700 mb-3">
              NO embedding needed!<br/>
              Use pre-computed STY vocab<br/>
              Vectorized similarity
            </div>
            <div className="text-sm text-gray-600">Intuition: Type-safety prior; prevents meds matching to diagnoses, etc.</div>
            <div className="bg-gray-100 p-3 rounded font-mono text-sm">
              sty_score = sty_map.get(candidate_STY)
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 text-white p-8 rounded-2xl">
        <h3 className="text-3xl font-bold mb-6">Search Statistics</h3>
        <div className="grid grid-cols-4 gap-6 text-center">
          <div>
            <div className="text-4xl font-bold mb-2">4</div>
            <div className="text-gray-400">Signals</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2">2</div>
            <div className="text-gray-400">Systems</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2">500</div>
            <div className="text-gray-400">Top-K each</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2">~4K</div>
            <div className="text-gray-400">Total candidates</div>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-2xl border-2 border-gray-200 mt-6">
        <h3 className="text-2xl font-bold mb-4">FAISS Implementation Notes</h3>
        <ul className="list-disc pl-6 text-gray-700 space-y-1">
          <li>HNSW (Flat-IP): M=32, efConstruction=200, efSearch=128 (tune for recall/latency)</li>
          <li>L2-normalized vectors so inner product equals cosine</li>
          <li>Variants: IVFPQ (trained, low memory), Flat-IP (exact, high memory)</li>
        </ul>
      </div>

      <div className="grid grid-cols-1 gap-4 mt-2">
        <button onClick={() => onDetail('semantic-search')} className="w-full py-4 bg-blue-600 text-white text-xl font-bold rounded-xl hover:bg-blue-700">
          FAISS Implementation Details →
        </button>
      </div>
    </div>
  )
}
