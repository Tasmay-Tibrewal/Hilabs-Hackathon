export const testExamples = {
  procedures: [
    { input: "polypectomy", output: "Polypectomy", code: "82035006", system: "SNOMED", tty: "SY" },
    { input: "right salpingectomy", output: "Right salpingectomy", code: "176916000", system: "SNOMED", tty: "PT" },
    { input: "coronary angioplasty", output: "Coronary angioplasty", code: "41339005", system: "SNOMED", tty: "PT" },
    { input: "endometrial biopsy", output: "Endometrial biopsy", code: "386802000", system: "SNOMED", tty: "PT" },
  ],
  medications: [
    { input: "doxycycline hyclate 100 mg capsule", output: "doxycycline hyclate 100 MG Oral Capsule", code: "1649988", system: "RXNORM", tty: "SCD" },
    { input: "ibuprofen 600 mg oral tablet", output: "ibuprofen 600 MG Oral Tablet", code: "197806", system: "RXNORM", tty: "PSN" },
    { input: "atorvastatin 10 mg", output: "atorvastatin 10 MG", code: "597970", system: "RXNORM", tty: "SCDC" },
    { input: "acetaminophen 325 mg tablet", output: "acetaminophen 325 MG Oral Tablet", code: "313782", system: "RXNORM", tty: "PSN" },
  ],
  labs: [
    { input: "hemoglobin alc", output: "Hemoglobin Russ", code: "54434005", system: "SNOMED", tty: "PT" },
    { input: "urinalysis w/reflex microscopic", output: "Urinalysis with reflex to microscopy", code: "442468009", system: "SNOMED", tty: "PT" },
  ],
  diagnoses: [
    { input: "allergic rhinitis", output: "Allergic rhinitis", code: "61582004", system: "SNOMED", tty: "PT" },
    { input: "tinea corporis", output: "Tinea corporis", code: "84849002", system: "SNOMED", tty: "PT" },
  ]
}

export const detailContent = {
  'dense-retrieval': {
    title: "Dense Retrieval: FAISS + Embeddings",
    sections: [
      {
        heading: "Embedding Model: google/embeddinggemma-300m",
        content: [
          "Dimensions: 384 (compact but powerful)",
          "All vectors L2-normalized to unit length",
          "Inner product = cosine similarity on normalized vectors",
          "Converts to [0,1]: cos_to_01(x) = clip((x+1)/2, 0, 1)"
        ]
      },
      {
        heading: "FAISS Index: HNSW (Default)",
        content: [
          "Hierarchical Navigable Small World graph",
          "M=32: connections per node",
          "efConstruction=200: build-time accuracy",
          "efSearch=128: query-time breadth",
          "Search complexity: O(log N) average",
          "Recall: >95% with proper tuning"
        ]
      },
      {
        heading: "Memory-Mapped Vectors (*.f32)",
        content: [
          "Why store separately? Exact embeddings without lossy reconstruction",
          "snomed_vectors.f32: 900K × 384 × 4 bytes ≈ 1.4GB",
          "rxnorm_vectors.f32: 500K × 384 × 4 bytes ≈ 0.8GB",
          "Row-aligned: row_id directly indexes array",
          "Enables fast vectorized per-code aggregation"
        ]
      }
    ]
  },
  'fuzzy-matching': {
    title: "Two-Stage Fuzzy Matching",
    sections: [
      {
        heading: "Why Two Stages?",
        content: [
          "Need BOTH speed (500+ candidates) AND accuracy (token-level matching)",
          "Stage 1 (ratio): Fast prefilter, eliminates ~60-80% of poor matches",
          "Stage 2 (token_set_ratio): Accurate refinement on reduced set"
        ]
      },
      {
        heading: "Anchors Concept",
        content: [
          "Multiple reference strings from: {query} ∪ LLM keywords",
          "Example: 'paracetamol 500' → anchors: [paracetamol, acetaminophen, tylenol, panadol]",
          "Final score = MAX across all anchor similarities",
          "Typically 3-8 anchors per query"
        ]
      }
    ]
  },
  // Add more detail content for all buttons...
}