# Website Structure and Editing Guide

This document explains how the presentation website is organized, how to edit slides and deep‑dives, and common tips for Mermaid/JSX.

## Overview

- Framework: Next.js 14 (App Router), TypeScript, Tailwind CSS
- Rendering: Static export (no server), Mermaid via CDN (client‑only)
- Interaction: 20 slides; deep‑dives appear in full‑screen popups
- Navigation: ←/→ arrows, Space to advance, Esc closes popups

## Runtime entry

- `app/page.tsx` renders `presentation-page.tsx`
- `app/presentation-page.tsx` orchestrates:
  - 20 slides with progress bar and dots
  - Keyboard handlers
  - Popups via `DetailSlide`

## Slides (modular)

```
app/slides/
  slides-0-4.tsx      # 0–4: Title, Problem, Baselines, Hybrid
  slide-6.tsx         # 6: Semantic Search (4 signals)
  slides-5-8.tsx      # 5: Build, 7: Deterministic Mode, 8: Multi‑Signal
  slides-9-12.tsx     # 9: STY, 10: Two‑Stage Fuzzy, 11: Aggregation, 12: Architecture
  slides-13-16.tsx    # 13: Concurrency, 14: Memory, 15: Rerank, 16: Results
  slides-17-19.tsx    # 17: Future, 18: Tech Stack, 19: Summary
```

## Deep‑dives

All popup content is declared in one place:

```
app/data/deep-dives.tsx
```

Each deep‑dive is indexed by an ID. Slides open a deep‑dive by calling `onDetail('id')`.

Key IDs (non‑exhaustive):

- `method-faiss`, `method-fuzzy`, `method-llm`, `method-scoring`
- `build-load`, `build-schema`, `build-embed`, `build-faiss`, `build-memmap`, `build-sty`
- `pipeline-detail`, `semantic-search`, `llm-expansion`, `scoring-math`, `sty-deep`, `fuzzy-algo`, `aggregation`
- `concurrency`, `memory`, `non-llm`, `backends`, `arch-diagrams`
- `future-hnsw`, `future-rl`, `future-desc`, `future-bm25`, `future-cluster`, `future-cache`

### Adding a new deep‑dive

1. Open `app/data/deep-dives.tsx`
2. Add a new entry with `{ title, content }` following the existing visual pattern
3. From a slide, wire a button to `onDetail('your-id')`

### Visual guidelines

- Use color‑coded sections:
  - bg‑blue‑50: technical details
  - bg‑green‑50: positives/solutions
  - bg‑yellow‑50: cautions/notes
  - bg‑red‑50: problems/failures
  - bg‑purple‑50: advanced topics
- Content cards: `bg-white p-6 rounded-lg`
- Code blocks: `bg-gray-900 text-green-400 font-mono`
- Diagrams via `<MermaidDiagram chart={'...'} />`

## Mermaid tips

- Keep node labels ASCII‑safe; avoid curly quotes or Unicode arrows; prefer plain text node IDs (e.g., `LoadParquet` not `"Load Parquet"`).
- Escape visible JSX arrows in text with `-&gt;` when needed.
- Mermaid initialization happens client‑side in `components/MermaidDiagram.tsx`.

## Running & building

```
npm install
npm run dev       # http://localhost:3000
npm run build     # outputs ./out
```

## Deployment

- GitHub Pages workflow is included at `website/.github/workflows/deploy.yml`
- Or serve `./out` on any static host (Netlify, S3, etc.)

## Contributing pattern

1. Update slides in `app/slides/` using concise blocks and clear 1–2 line descriptions
2. Put heavy technical detail in deep‑dives; keep slides scannable
3. When adding new diagrams/snippets, prefer reusing existing section styles for consistency

