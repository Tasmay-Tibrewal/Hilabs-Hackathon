# Clinical Concept Harmonizer — Presentation Website

Static, single‑page presentation (20 slides) with interactive deep‑dives.

## Quick start

```
npm install
npm run dev     # http://localhost:3000
```

Build static
```
npm run build   # output in ./out
```

## Tech & features
- Next.js 14, TypeScript, Tailwind CSS
- Mermaid.js via CDN (client‑only)
- Keyboard: ←/→, Space; Esc closes deep dives
- Popups contain code blocks, Mermaid diagrams, and color‑coded sections

## App structure
```
app/
  presentation-page.tsx   # 20‑slide controller with keyboard & progress
  slides/
    slides-0-4.tsx        # 0–4: Title, Problem, Baselines, Hybrid
    slide-6.tsx           # 6: Semantic Search (4 signals)
    slides-5-8.tsx        # 5: Build, 7: Deterministic Mode, 8: Multi‑Signal
    slides-9-12.tsx       # 9: STY, 10: Fuzzy, 11: Aggregation, 12: Architecture
    slides-13-16.tsx      # 13: Concurrency, 14: Memory, 15: Rerank, 16: Results
    slides-17-19.tsx      # 17: Future, 18: Tech Stack, 19: Summary
  data/
    deep-dives.tsx        # All deep‑dive popup content
  components/
    MermaidDiagram.tsx    # Client‑only Mermaid renderer
    DetailSlide.tsx       # Full‑screen popup container
  layout.tsx              # Mermaid CDN + global shell
  page.tsx                # Entry that renders presentation‑page
globals.css               # Tailwind styles
```

## Editing deep‑dives
- Edit `app/data/deep-dives.tsx`; follow existing patterns (bg‑*‑50 sections, code, tables, Mermaid).
- Use ASCII‑safe labels in Mermaid nodes; escape JSX arrows as `-&gt;` in text when needed.

## Deploy
- GitHub Pages workflow under `.github/workflows/deploy.yml` (uses static `out/`).
- Or serve `./out` on any static host.

## More
- See WEBSITE_STRUCTURE.md for slide mapping, deep‑dive IDs, Mermaid tips, and editing guide.
