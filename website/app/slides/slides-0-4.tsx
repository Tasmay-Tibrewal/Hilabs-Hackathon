// Slides 0-4: Title, Problem, Examples, Why Fail, Hybrid Solution

export function Slide0() {
  return (
    <div className="max-w-6xl text-center">
      <div className="text-sm font-mono text-gray-500 mb-6">HiLabs Hackathon 2025</div>
      <h1 className="text-8xl font-bold mb-8 leading-tight">
        Clinical Concept<br/><span className="bg-black text-white px-6">Harmonizer</span>
      </h1>
      <p className="text-3xl text-gray-600 mb-12 max-w-4xl mx-auto">
        Hybrid AI System for Clinical Data Normalization<br/>
        <strong>RxNorm</strong> + <strong>SNOMED CT</strong>
      </p>
      <div className="grid grid-cols-3 gap-6 max-w-4xl mx-auto">
        <div className="bg-gray-50 p-6 rounded-xl">
          <div className="text-4xl font-bold mb-2">~1.4M</div>
          <div className="text-gray-600">Catalog Entries</div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <div className="text-4xl font-bold mb-2">~600K</div>
          <div className="text-gray-600">Unique Codes</div>
        </div>
        <div className="bg-gray-50 p-6 rounded-xl">
          <div className="text-4xl font-bold mb-2">4-Method</div>
          <div className="text-gray-600">Hybrid AI</div>
        </div>
      </div>
      <div className="mt-12 text-xl text-gray-500 font-mono">Press → or SPACE</div>
    </div>
  )
}

export function Slide1({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-12">The Problem: Data Chaos</h2>
      
      <div className="grid grid-cols-2 gap-8 mb-8">
        <div className="bg-red-50 border-4 border-red-300 p-8 rounded-2xl">
          <h3 className="text-3xl font-bold mb-6 text-red-900">What Hospitals Send</h3>
          <div className="space-y-4">
            {["chest xr", "CXR", "chest x-ray", "xr chest portable"].map((t, i) => (
              <div key={i} className="bg-white p-4 rounded-lg border-2 border-red-200">
                <div className="font-mono text-xl text-red-600 mb-1">"{t}"</div>
                <div className="text-sm text-gray-600">Hospital {String.fromCharCode(65 + i)}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-green-50 border-4 border-green-300 p-8 rounded-2xl">
          <h3 className="text-3xl font-bold mb-6 text-green-900">What We Need</h3>
          <div className="bg-white p-6 rounded-lg border-4 border-green-500">
            <div className="text-sm text-gray-500 mb-3">STANDARDIZED OUTPUT</div>
            <div className="text-2xl font-bold mb-4">Plain X-ray of chest</div>
            <div className="space-y-2">
              <div><span className="px-3 py-1 bg-green-100 text-green-900 rounded font-mono">SNOMED: 399208008</span></div>
              <div><span className="px-3 py-1 bg-gray-100 rounded">STY: Diagnostic Procedure</span></div>
              <div><span className="px-3 py-1 bg-gray-100 rounded">TTY: PT</span></div>
            </div>
          </div>
          <div className="mt-6 text-center text-2xl font-bold text-green-700">One consistent code!</div>
        </div>
      </div>

      <div className="bg-gray-900 text-white p-8 rounded-2xl">
        <h4 className="text-2xl font-bold mb-4">Why This Matters</h4>
        <div className="grid grid-cols-2 gap-6 text-xl">
          <div>❌ Cannot aggregate data across hospitals</div>
          <div>❌ Analytics systems break</div>
          <div>❌ Regulatory reporting fails</div>
          <div>❌ Clinical research stalls</div>
        </div>
      </div>
    </div>
  )
}

export function Slide2() {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Real Examples from Test Data</h2>
      <p className="text-2xl text-gray-600 mb-8">Actual outputs from Test_with_predictions.xlsx</p>
      
      <div className="space-y-4">
        {[
          { input: "polypectomy", type: "Procedure", output: "Polypectomy", code: "82035006", system: "SNOMED", color: "green" },
          { input: "doxycycline hyclate 100 mg capsule", type: "Medicine", output: "doxycycline hyclate 100 MG Oral Capsule", code: "1649988", system: "RXNORM", color: "blue" },
          { input: "right salpingectomy", type: "Procedure", output: "Right salpingectomy", code: "176916000", system: "SNOMED", color: "green" },
          { input: "allergic rhinitis", type: "Diagnosis", output: "Allergic rhinitis", code: "61582004", system: "SNOMED", color: "green" },
        ].map((ex, i) => (
          <div key={i} className={`bg-gradient-to-r from-${ex.color}-50 to-white border-l-8 border-${ex.color}-500 p-6 rounded-r-2xl`}>
            <div className="grid grid-cols-5 gap-4 items-center">
              <div className="col-span-2">
                <div className="text-xs text-gray-500 mb-1">INPUT</div>
                <div className="font-mono font-bold text-xl">{ex.input}</div>
                <div className="text-gray-600 mt-1">Type: <strong>{ex.type}</strong></div>
              </div>
              <div className="text-center text-3xl">→</div>
              <div className="col-span-2">
                <div className="text-xs text-gray-500 mb-1">OUTPUT</div>
                <div className="font-bold text-lg">{ex.output}</div>
                <div className="mt-2"><span className={`px-2 py-1 bg-${ex.color}-100 text-${ex.color}-900 rounded font-mono text-sm`}>{ex.system}: {ex.code}</span></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 bg-black text-white p-6 rounded-2xl text-center">
        <div className="text-2xl font-bold">All from real test data - verified correct mappings</div>
      </div>
    </div>
  )
}

export function Slide3({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-8">Why Simple Approaches Fail</h2>
      
      <div className="space-y-6">
        <div className="bg-red-50 border-4 border-red-300 p-8 rounded-2xl">
          <div className="flex items-start gap-6">
            <div className="text-6xl">❌</div>
            <div className="flex-1">
              <h3 className="text-3xl font-bold mb-3 text-red-900">String Matching Alone</h3>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-white p-4 rounded-lg">
                  <div className="font-mono mb-2">"paracetamol" vs "Acetaminophen"</div>
                  <div className="text-red-600 font-bold">0% similarity!</div>
                </div>
                <div className="bg-white p-4 rounded-lg">
                  <div className="font-mono mb-2">"chest xr" vs "Plain X-ray"</div>
                  <div className="text-red-600 font-bold">Abbreviation mismatch</div>
                </div>
              </div>
              <button onClick={() => onDetail('string-fail')} className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-bold">
                Why It Fails →
              </button>
            </div>
          </div>
        </div>

        <div className="bg-red-50 border-4 border-red-300 p-8 rounded-2xl">
          <div className="flex items-start gap-6">
            <div className="text-6xl">❌</div>
            <div className="flex-1">
              <h3 className="text-3xl font-bold mb-3 text-red-900">Pure Embeddings Alone</h3>
              <ul className="space-y-2 text-xl text-gray-700 mb-4">
                <li>• Short queries ("Hb", "xr") have poor context</li>
                <li>• No domain knowledge (brands, abbreviations)</li>
                <li>• Cannot distinguish similar conditions</li>
              </ul>
              <button onClick={() => onDetail('embedding-fail')} className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-bold">
                Technical Details →
              </button>
            </div>
          </div>
        </div>

        <div className="bg-red-50 border-4 border-red-300 p-8 rounded-2xl">
          <div className="flex items-start gap-6">
            <div className="text-6xl">❌</div>
            <div className="flex-1">
              <h3 className="text-3xl font-bold mb-3 text-red-900">Pure LLM Alone</h3>
              <ul className="space-y-2 text-xl text-gray-700 mb-4">
                <li>• <strong>Hallucination:</strong> Invents fake codes</li>
                <li>• No access to 1.4M catalog</li>
                <li>• Expensive at scale</li>
              </ul>
              <button onClick={() => onDetail('llm-fail')} className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-bold">
                See Examples →
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 bg-black text-white p-8 rounded-2xl text-center">
        <div className="text-3xl font-bold">→ Need a Hybrid Approach</div>
      </div>
    </div>
  )
}

export function Slide4({ onDetail }: { onDetail: (id: string) => void }) {
  return (
    <div className="max-w-6xl w-full">
      <h2 className="text-6xl font-bold mb-12">Our 4-Method Hybrid Solution</h2>
      
      <div className="grid gap-6">
        {[
          { num: 1, title: "Dense Retrieval", desc: "FAISS + SentenceTransformers over 1.4M entries", color: "blue", id: "method-faiss" },
          { num: 2, title: "Fuzzy Matching", desc: "Two-stage RapidFuzz: ratio → token_set_ratio", color: "green", id: "method-fuzzy" },
          { num: 3, title: "LLM Intelligence", desc: "Qwen3-4B expansion + optional rerank", color: "purple", id: "method-llm" },
          { num: 4, title: "Multi-Signal Scoring", desc: "30% desc + 40% kw + 20% direct + 10% STY", color: "orange", id: "method-scoring" },
        ].map(m => (
          <div key={m.num} className={`bg-gradient-to-r from-${m.color}-50 to-white border-l-8 border-${m.color}-500 p-8 rounded-r-2xl`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6 flex-1">
                <div className={`text-6xl font-bold text-${m.color}-600`}>{m.num}</div>
                <div>
                  <h3 className="text-3xl font-bold mb-2">{m.title}</h3>
                  <p className="text-xl text-gray-700">{m.desc}</p>
                  <p className="text-gray-600 mt-2">
                    {m.id === 'method-faiss' && 'Semantic neighbors across both vocabularies; compact 384-d embeddings enable fast and accurate retrieval.'}
                    {m.id === 'method-fuzzy' && 'Lexical safety net for abbreviations, brands, and token reordering; anchors from query ∪ LLM keywords.'}
                    {m.id === 'method-llm' && 'Adds domain knowledge to short inputs; final rerank picks among provided codes only (no hallucination).'}
                    {m.id === 'method-scoring' && 'Blend signals to reduce brittleness; any one strong signal can carry the result while others corroborate.'}
                  </p>
                </div>
              </div>
              <button onClick={() => onDetail(m.id === 'method-fuzzy' ? 'fuzzy-deep' : m.id)} className={`px-6 py-3 bg-${m.color}-600 text-white rounded-lg hover:bg-${m.color}-700 font-bold ml-4`}>
                {m.id === 'method-fuzzy' ? 'Fuzzy Matching Deep Dive →' : 'Deep Dive →'}
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 bg-black text-white p-8 rounded-2xl text-center">
        <div className="text-3xl font-bold mb-3">Why Hybrid Works</div>
        <div className="text-xl text-gray-300">Semantic + Lexical + Domain Knowledge + Type Safety</div>
      </div>
    </div>
  )
}
