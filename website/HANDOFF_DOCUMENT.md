# Clinical Concept Harmonizer - Website Development Handoff Document

**Date:** January 7, 2025  
**From:** Kilo Code (Claude Sonnet 4.5)  
**To:** Codex Agent  
**Project:** HiLabs Hackathon 2025 - Presentation Website

---

## 1. PROJECT OVERVIEW

### 1.1 What This Is
A Next.js-based presentation website (20 slides) showcasing the Clinical Concept Harmonizer solution for the HiLabs Hackathon 2025. The solution maps messy clinical data to standardized RxNorm (medications) and SNOMED CT (diagnoses/procedures/labs) codes using a hybrid AI approach.

### 1.2 Tech Stack
- **Framework:** Next.js 14.0.4 (React 18.2.0)
- **Language:** TypeScript 5.3.3
- **Styling:** Tailwind CSS 3.4.0
- **Deployment:** Static export for GitHub Pages/Vercel
- **Diagrams:** Mermaid.js (CDN-loaded, client-side only)
- **Design:** Monochrome (black/white/gray), professional like OpenAI/Uber

### 1.3 Key Requirements
- 20-slide presentation format (not scrollable website)
- Keyboard navigation (arrow keys, SPACE)
- Detail popup system for technical deep dives
- Real test data only (NO fake metrics/demos)
- Modular architecture to avoid file size issues
- All markdown links must be clickable with proper syntax

---

## 2. COMPLETED WORK

### 2.1 Project Structure Created ✅

```
website/
├── app/
│   ├── layout.tsx              # Root layout with Mermaid CDN
│   ├── page.tsx                # Entry point (routes to presentation-page)
│   ├── globals.css             # Tailwind + custom styles
│   ├── presentation-page.tsx   # Main presentation controller (93 lines)
│   │
│   ├── components/
│   │   ├── MermaidDiagram.tsx  # Client-side diagram renderer (58 lines)
│   │   └── DetailSlide.tsx     # Full-screen popup component (37 lines)
│   │
│   ├── slides/                 # MODULAR SLIDE FILES
│   │   ├── slides-0-4.tsx      # Slides 0-4 (98 lines)
│   │   ├── slide-6.tsx         # Slide 6 (77 lines)
│   │   ├── slides-5-8.tsx      # Slides 5,7,8 (111 lines)
│   │   ├── slides-9-12.tsx     # Slides 9-12 (116 lines)
│   │   ├── slides-13-16.tsx    # Slides 13-16 (119 lines)
│   │   └── slides-17-19.tsx    # Slides 17-19 (114 lines)
│   │
│   └── data/
│       ├── presentation-content.ts  # Real test examples from CSV
│       └── deep-dives.tsx          # Rich popup content (253 lines - INCOMPLETE)
│
├── public/                     # Static assets (if needed)
├── .github/workflows/
│   └── deploy.yml              # Auto-deployment to GitHub Pages
│
├── package.json                # Dependencies
├── tsconfig.json               # TypeScript config
├── tailwind.config.ts          # Tailwind with gray palette
├── postcss.config.js           # PostCSS for Tailwind
├── next.config.js              # Static export config
├── .gitignore                  # Standard Next.js ignores
│
└── Documentation/
    ├── README.md               # Main docs (199 lines)
    ├── DEPLOYMENT.md           # Deployment guide (294 lines)
    ├── SETUP.md                # Setup guide (343 lines)
    ├── PRESENTATION.md         # Presentation script (246 lines)
    └── HANDOFF_DOCUMENT.md     # This file
```

### 2.2 Core Components Explained

#### 2.2.1 `presentation-page.tsx` (Main Controller)
**Purpose:** Manages the entire presentation
**Key Features:**
- State management: `currentSlide` (0-19)
- Keyboard navigation: ArrowLeft, ArrowRight, Space
- Detail popup system
- Progress bar (bottom)
- Navigation dots (right side)
- Imports all slide modules

**How It Works:**
```typescript
// State
const [currentSlide, setCurrentSlide] = useState(0)
const [detailSlide, setDetailSlide] = useState<string | null>(null)

// Navigation
const nextSlide = () => setCurrentSlide(prev => Math.min(19, prev + 1))
const prevSlide = () => setCurrentSlide(prev => Math.max(0, prev - 1))

// Renders current slide from imported modules
{currentSlide === 0 && <Slide0 onDetailClick={setDetailSlide} />}
{currentSlide === 1 && <Slide1 onDetailClick={setDetailSlide} />}
// ... etc for all 20 slides

// Detail popup
{detailSlide && deepDives[detailSlide] && (
  <DetailSlide
    title={deepDives[detailSlide].title}
    content={deepDives[detailSlide].content}
    onClose={() => setDetailSlide(null)}
  />
)}
```

#### 2.2.2 `MermaidDiagram.tsx` (Diagram Renderer)
**Purpose:** Render Mermaid diagrams client-side only (avoiding hydration errors)

**Problem Solved:** Next.js SSR tries to render diagrams on server, but Mermaid only works in browser. This caused hydration mismatches.

**Solution:**
```typescript
'use client'
import { useEffect, useState } from 'react'

export default function MermaidDiagram({ chart }: { chart: string }) {
  const [svg, setSvg] = useState<string>('')
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)  // Only render on client
    if (typeof window !== 'undefined' && window.mermaid) {
      window.mermaid.render('mermaid-' + Math.random(), chart)
        .then(result => setSvg(result.svg))
    }
  }, [chart])

  if (!isClient) return <div>Loading diagram...</div>
  return <div dangerouslySetInnerHTML={{ __html: svg }} />
}
```

#### 2.2.3 `DetailSlide.tsx` (Popup Component)
**Purpose:** Full-screen overlay for technical deep dives

**Features:**
- Fixed overlay with semi-transparent background
- Click outside to close
- ESC key to close
- Scrollable content area
- Close button (×) in top-right

**Usage in slides:**
```typescript
<button 
  onClick={() => onDetailClick('string-fail')}
  className="text-blue-600 hover:underline"
>
  Deep Dive →
</button>
```

### 2.3 Slide Architecture (Modular System)

**Why Modular?** 
- Large single files (>3000 lines) caused corruption
- Easier maintenance and debugging
- Clear separation of concerns
- Avoids token limits

**Slide Distribution:**

| File | Slides | Content | Lines |
|------|--------|---------|-------|
| `slides-0-4.tsx` | 0-4 | Title, Problem, Examples, Failures, Hybrid | 98 |
| `slide-6.tsx` | 6 | Semantic Search (4 signals) | 77 |
| `slides-5-8.tsx` | 5,7,8 | Build, LLM Expansion, Scoring | 111 |
| `slides-9-12.tsx` | 9-12 | STY, Fuzzy, Aggregation, Architecture | 116 |
| `slides-13-16.tsx` | 13-16 | Concurrency, Memory, Rerank, Results | 119 |
| `slides-17-19.tsx` | 17-19 | Future Work, Tech Stack, Summary | 114 |

**Each slide module exports:**
```typescript
export function Slide0({ onDetailClick }: SlideProps) {
  return (
    <div className="h-full flex flex-col items-center justify-center p-12">
      {/* Slide content */}
    </div>
  )
}

export function Slide1({ onDetailClick }: SlideProps) { /* ... */ }
// etc.
```

**SlideProps interface:**
```typescript
interface SlideProps {
  onDetailClick: (detailId: string) => void
}
```

### 2.4 Configuration Files

#### 2.4.1 `next.config.js`
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',           // Static HTML export
  images: { unoptimized: true }, // No Image Optimization API
  distDir: 'out',             // Output directory
  trailingSlash: true,        // Add trailing slashes
}
module.exports = nextConfig
```

#### 2.4.2 `tailwind.config.ts`
```typescript
module.exports = {
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          // ... full gray scale for monochrome theme
        }
      }
    }
  }
}
```

#### 2.4.3 `package.json` Dependencies
```json
{
  "dependencies": {
    "next": "14.0.4",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1",
    "postcss": "^8",
    "tailwindcss": "^3.4.0",
    "typescript": "^5"
  },
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  }
}
```

### 2.5 Styling System

**Color Palette (Monochrome):**
- Background: `bg-white`
- Text: `text-black`, `text-gray-700`, `text-gray-600`
- Accents: `bg-gray-100`, `bg-gray-200`
- Borders: `border-gray-300`
- Hover states: `hover:bg-gray-100`

**Key Classes:**
```css
/* Slide container */
.h-full .flex .flex-col .items-center .justify-center .p-12

/* Cards */
.bg-white .p-6 .rounded-xl .border-4 .border-gray-200

/* Buttons */
.px-6 .py-3 .bg-black .text-white .rounded-lg .hover:bg-gray-800

/* Progress bar */
.fixed .bottom-0 .left-0 .h-2 .bg-black .transition-all

/* Navigation dots */
.fixed .right-8 .top-1/2 .-translate-y-1/2 .space-y-3
```

### 2.6 Navigation System

**Keyboard Controls:**
```typescript
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if (detailSlide) {
      if (e.key === 'Escape') setDetailSlide(null)
      return
    }
    
    switch(e.key) {
      case 'ArrowRight':
      case ' ':  // Space key
        nextSlide()
        break
      case 'ArrowLeft':
        prevSlide()
        break
    }
  }
  
  window.addEventListener('keydown', handleKeyDown)
  return () => window.removeEventListener('keydown', handleKeyDown)
}, [detailSlide, currentSlide])
```

**Visual Indicators:**
- Progress bar: Shows `(currentSlide / 19) * 100%` width
- Navigation dots: 20 dots, current one is larger and darker
- Slide counter: "Slide X of 20" in top-right

---

## 3. INCOMPLETE WORK (CRITICAL)

### 3.1 Missing Deep Dive Content

**Current State:**
`deep-dives.tsx` has only 3 deep dives implemented:
- ✅ `string-fail` (Why String Matching Fails)
- ✅ `embedding-fail` (Pure Embeddings Limitations)
- ✅ `llm-fail` (Pure LLM Critical Failures)

**Missing Deep Dives (Referenced in Slides):**

#### From Slide 4 (Hybrid Methods):
- `method-faiss` - Dense Retrieval: FAISS + Embeddings
- `method-fuzzy` - Fuzzy Matching: Two-Stage Algorithm
- `method-llm` - LLM Intelligence details
- `method-scoring` - Multi-Signal Scoring details

#### From Slide 5 (Build Phase):
- `build-load` - Loading parquet files
- `build-schema` - Schema normalization
- `build-embed` - Embedding generation
- `build-faiss` - FAISS index creation
- `build-memmap` - Memory-mapped vectors
- `build-sty` - STY vocabulary embeddings

#### From Other Slides:
- `scoring-math` - Multi-Signal Scoring mathematics (referenced in Slide 8)
- `aggregation` - Per-Code Aggregation formula (referenced in Slide 11)
- `llm-expansion` - LLM Query Expansion (referenced in Slide 7)
- `sty-deep` - STY Compatibility implementation (referenced in Slide 9)
- `fuzzy-algo` - Fuzzy algorithm pseudocode (referenced in Slide 10)
- `concurrency` - Concurrency implementation (referenced in Slide 13)
- `memory` - Memory management strategy (referenced in Slide 14)
- `llm-rerank` - LLM reranking details (referenced in Slide 15)

#### From Slide 17 (Future Work):
- `future-hnsw` - Enhanced HNSW with small-world connections
- `future-rl` - GRPO Reinforcement Learning pipeline
- `future-desc` - Build-time rich descriptions
- `future-bm25` - BM25 lexical matching
- `future-cluster` - Clustering techniques
- `future-cache` - Caching expansions

### 3.2 Deep Dive Content Structure

**Each deep dive should follow this pattern:**
```typescript
'detail-id': {
  title: 'Descriptive Title',
  content: (
    <div className="space-y-6">
      {/* Section 1 */}
      <div className="bg-blue-50 p-6 rounded-xl">
        <h4 className="text-3xl font-bold mb-4">Section Title</h4>
        <div className="bg-white p-6 rounded-lg">
          {/* Content with grids, code blocks, examples */}
        </div>
      </div>

      {/* Section 2 */}
      <div className="bg-green-50 p-6 rounded-xl">
        {/* More visual content */}
      </div>

      {/* Use color-coded sections: */}
      {/* - bg-blue-50: Technical details */}
      {/* - bg-green-50: Positive/solutions */}
      {/* - bg-yellow-50: Warnings/notes */}
      {/* - bg-red-50: Problems/failures */}
      {/* - bg-purple-50: Advanced topics */}
    </div>
  )
}
```

**Design Guidelines:**
- Use large headings (text-3xl, text-2xl)
- Color-code sections with bg-*-50 backgrounds
- White cards for content: bg-white p-6 rounded-lg
- Code blocks: bg-gray-900 text-green-400 font-mono
- Grids for comparisons: grid grid-cols-2 gap-6
- Tables for data
- Lists with proper spacing

### 3.3 Content Source for Deep Dives

**All technical content is in the original files:**

1. **`task_prompt.txt`** - Contains:
   - Problem statement details
   - Sample input/output examples
   - Evaluation criteria
   - Column reference guide

2. **`README.md`** (in repo root) - Contains:
   - Architecture overview
   - How it works
   - Quickstart commands
   - Tuning parameters
   - Methodology details

3. **`Arch.md`** (in repo root) - Contains:
   - Deep technical architecture
   - Algorithm pseudocode
   - Mathematical formulas
   - Performance characteristics
   - Concurrency model
   - Memory management

4. **`clean.py`** - Contains:
   - Actual implementation
   - Comments explaining algorithms
   - Parameter defaults
   - Helper functions

**Use these to create rich, detailed deep dives with:**
- Real code snippets
- Mathematical formulas
- Concrete examples
- Performance numbers
- Configuration parameters

---

## 4. HOW TO ADD MISSING DEEP DIVES

### 4.1 Step-by-Step Process

**Step 1: Read Source Material**
```bash
# Read these files from repo root:
- task_prompt.txt
- README.md
- Arch.md
- clean.py (sections relevant to the feature)
```

**Step 2: Create Deep Dive Entry**
Open `website/app/data/deep-dives.tsx` and add before the closing `}`

**Step 3: Follow This Template**
```typescript
'new-detail-id': {
  title: 'Clear Descriptive Title',
  content: (
    <div className="space-y-6">
      {/* Introduction Section */}
      <div className="bg-blue-50 p-6 rounded-xl">
        <h4 className="text-3xl font-bold mb-4">What Is This?</h4>
        <div className="bg-white p-6 rounded-lg">
          <p className="text-xl mb-4">Brief explanation...</p>
          {/* Add visual elements */}
        </div>
      </div>

      {/* Technical Details Section */}
      <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
        <h4 className="text-3xl font-bold mb-4">Technical Implementation</h4>
        {/* Code blocks, formulas, diagrams */}
      </div>

      {/* Example Section */}
      <div className="bg-green-50 p-6 rounded-xl">
        <h4 className="text-3xl font-bold mb-4">Concrete Example</h4>
        {/* Real examples with input/output */}
      </div>

      {/* Advanced/Future Section (optional) */}
      <div className="bg-purple-50 p-6 rounded-xl">
        <h4 className="text-3xl font-bold mb-4">Advanced Topics</h4>
        {/* Deep technical details */}
      </div>
    </div>
  )
},  // ← Don't forget trailing comma!
```

**Step 4: Test**
```bash
# Save file and check terminal for errors
# If error appears, check:
# - Balanced JSX tags (<div> has </div>)
# - Trailing commas after each entry
# - Proper escaping of special characters in JSX
```

### 4.2 Common Patterns to Use

**Code Block:**
```typescript
<pre className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-x-auto">
{`def function_name():
    # Code here
    return result`}
</pre>
```

**Formula Display:**
```typescript
<div className="bg-gradient-to-r from-blue-50 to-purple-50 p-8 rounded-2xl">
  <div className="text-center font-mono text-4xl font-bold">
    final = avg_all × avg_top × √log₁₀(% in pool)
  </div>
</div>
```

**Comparison Grid:**
```typescript
<div className="grid grid-cols-2 gap-6">
  <div className="bg-white p-4 rounded">
    <strong>Option A:</strong><br/>
    Details...
  </div>
  <div className="bg-white p-4 rounded">
    <strong>Option B:</strong><br/>
    Details...
  </div>
</div>
```

**Table:**
```typescript
<div className="overflow-x-auto">
  <table className="w-full text-left border-collapse">
    <thead>
      <tr className="bg-gray-100">
        <th className="p-3 border">Column 1</th>
        <th className="p-3 border">Column 2</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td className="p-3 border">Data 1</td>
        <td className="p-3 border">Data 2</td>
      </tr>
    </tbody>
  </table>
</div>
```

### 4.3 Example: How to Add `method-faiss`

```typescript
'method-faiss': {
  title: 'Dense Retrieval: FAISS + Embeddings Deep Dive',
  content: (
    <div className="space-y-6">
      <div className="bg-blue-50 p-6 rounded-xl">
        <h4 className="text-3xl font-bold mb-4">Embedding Model Details</h4>
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="font-bold">Model:</span>
                <code className="bg-gray-100 px-3 py-1 rounded">google/embeddinggemma-300m</code>
              </div>
              <div className="flex justify-between">
                <span className="font-bold">Dimensions:</span>
                <span className="font-mono">384</span>
              </div>
              <div className="flex justify-between">
                <span className="font-bold">Normalization:</span>
                <span>L2 (unit length)</span>
              </div>
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg">
            <div className="font-bold mb-3">Why This Model?</div>
            <ul className="space-y-2 text-gray-700">
              <li>• Compact: 384d vs 768d (BERT)</li>
              <li>• Fast inference (~5-10ms/text)</li>
              <li>• Good for medical domain</li>
              <li>• Open source (Apache 2.0)</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-xl border-4 border-gray-200">
        <h4 className="text-3xl font-bold mb-4">FAISS Index: HNSW</h4>
        <div className="space-y-4">
          <div className="font-bold text-xl">Parameters:</div>
          <div className="bg-gray-100 p-4 rounded font-mono text-sm">
            M = 32 (connections per node)<br/>
            efConstruction = 200 (build accuracy)<br/>
            efSearch = 128 (query breadth)
          </div>
          <div className="font-bold text-xl">Characteristics:</div>
          <ul className="space-y-2 text-gray-700">
            <li>• Search time: O(log N) average</li>
            <li>• Recall: &gt;95% with proper tuning</li>
            <li>• Graph-based navigation</li>
            <li>• Cache-friendly access</li>
          </ul>
        </div>
      </div>

      <div className="bg-green-50 p-6 rounded-xl">
        <h4 className="text-3xl font-bold mb-4">Inner Product = Cosine Similarity</h4>
        <div className="bg-white p-6 rounded-lg">
          <p className="text-xl mb-4">Since all vectors are L2-normalized:</p>
          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-xl">
            cos(A, B) = A·B / (||A|| × ||B||)<br/>
            <br/>
            When ||A|| = ||B|| = 1:<br/>
            cos(A, B) = A·B  (inner product!)
          </div>
        </div>
      </div>
    </div>
  )
},
```

---

## 5. KNOWN ISSUES & SOLUTIONS

### 5.1 Hydration Errors with Mermaid
**Problem:** Mermaid diagrams cause React hydration mismatch  
**Solution:** Use `MermaidDiagram` component which only renders client-side

### 5.2 File Size/Corruption
**Problem:** Large files (>3000 lines) get corrupted during writes  
**Solution:** Use modular slide architecture (3-5 slides per file max)

### 5.3 TypeScript/JSX Errors
**Problem:** Missing React import causes JSX parsing errors  
**Solution:** Always add `import React from 'react'` at top of .tsx files

### 5.4 Special Characters in JSX
**Problem:** Characters like `×`, `&`, `<`, `>` break JSX  
**Solution:** 
- Use HTML entities: `&times;`, `&amp;`, `&lt;`, `&gt;`
- Or wrap in `{` `}`: `{">"}`, `{"×"}`

### 5.5 Mermaid Syntax Errors
**Problem:** Special chars in Mermaid diagrams  
**Solution:** Wrap text with special chars in quotes: `["Plain X-ray"]`

---

## 6. TESTING CHECKLIST

### 6.1 Before Deployment
- [ ] Run `npm run dev` - No errors in terminal
- [ ] Open http://localhost:3000 - Page loads
- [ ] Navigate all 20 slides with arrow keys
- [ ] Test all "Deep Dive →" buttons
- [ ] Check all Mermaid diagrams render
- [ ] Test ESC key closes popups
- [ ] Test click-outside closes popups
- [ ] Verify responsive design (mobile/tablet/desktop)
- [ ] Check all links are clickable with proper paths
- [ ] Run `npm run build` - Static export succeeds
- [ ] Test built version: `npm start` or serve `out/` directory

### 6.2 Content Verification
- [ ] All slides have correct content
- [ ] No fake metrics or demo components
- [ ] Real test examples from Test_with_predictions.csv
- [ ] All technical terms accurate
- [ ] All code snippets valid
- [ ] All formulas correct

---

## 7. DEPLOYMENT INSTRUCTIONS

### 7.1 GitHub Pages (Recommended)

**Auto-deployment is configured** via `.github/workflows/deploy.yml`

**Manual Steps:**
1. Push all changes to GitHub repo
2. GitHub Actions will automatically build and deploy
3. Website will be live at: `https://username.github.io/repo-name/`

**OR Manual Deploy:**
```bash
cd website
npm run build
# Commit and push the 'out/' directory to gh-pages branch
```

### 7.2 Vercel (Alternative)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd website
vercel

# Follow prompts for project setup
```

### 7.3 Netlify (Alternative)

```bash
cd website
npm run build

# Drag and drop the 'out/' folder to Netlify dashboard
# OR connect GitHub repo for auto-deployment
```

---

## 8. PRIORITY TASKS FOR NEXT AGENT

### 8.1 CRITICAL (Must Complete)
1. **Add all missing deep dives** to `deep-dives.tsx` (see section 3.1)
   - Start with most referenced ones: `method-faiss`, `method-fuzzy`, `scoring-math`, `aggregation`
   - Use content from `Arch.md` and `README.md`
   - Follow template in section 4.3

2. **Test all deep dive buttons** work without errors
   - Click every "Deep Dive →" button
   - Verify popup opens with content
   - Check ESC and click-outside close properly

3. **Verify all Mermaid diagrams render**
   - Especially Slide 12 (architecture diagram)
   - Fix any syntax errors

### 8.2 HIGH PRIORITY
4. **Add more visual examples** to existing slides
   - Slide 2: Add 1-2 more real test cases
   - Slide 16: Show actual mapping results

5. **Enhance Future Work slide** (17)
   - Add more technical depth to each enhancement
   - Include expected impact metrics

### 8.3 MEDIUM PRIORITY
6. **Polish styling**
   - Ensure consistent spacing
   - Check font sizes are readable
   - Verify hover states work

7. **Add transitions** between slides (optional)
   - Fade in/out effects
   - Slide animations

### 8.4 LOW PRIORITY
8. **Add print/export functionality** (optional)
9. **Add slide notes** for presenter (optional)
10. **Add timer** for presentation timing (optional)

---

## 9. USEFUL COMMANDS

### 9.1 Development
```bash
cd website
npm install          # Install dependencies
npm run dev          # Start dev server (http://localhost:3000)
npm run build        # Build for production
npm start            # Start production server
```

### 9.2 Debugging
```bash
# Check TypeScript errors
npx tsc --noEmit

# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### 9.3 File Operations
```bash
# Create new slide file
touch app/slides/slide-new.tsx

# Check file size
ls -lh app/data/deep-dives.tsx

# Search for missing deep dives
grep -r "onDetailClick(" app/slides/
```

---

## 10. IMPORTANT NOTES

### 10.1 Design Philosophy
- **Progressive disclosure:** Start simple, expand on click
- **Visual hierarchy:** Use size, color, spacing to guide attention
- **Minimal color:** Black/white/gray only, use sparingly
- **Clear typography:** Large headings, readable body text
- **Generous spacing:** Don't crowd content

### 10.2 Content Guidelines
- **Real data only:** No invented metrics or fake demos
- **Technical accuracy:** All formulas and code must be correct
- **Concise:** Each slide should take 60-90 seconds to present
- **Progressive:** Basic → Detailed via deep dives
- **Visual:** Prefer diagrams/tables over text walls

### 10.3 Code Quality
- **TypeScript strict mode:** Use proper types
- **Component reusability:** Extract common patterns
- **Performance:** Lazy load heavy content
- **Accessibility:** ARIA labels, keyboard nav, contrast ratios

---

## 11. CONTACT & REFERENCES

### 11.1 Original Files (Repo Root)
- `task_prompt.txt` - Full problem statement
- `README.md` - Project overview
- `Arch.md` - Technical architecture
- `clean.py` - Implementation code

### 11.2 Documentation
- `website/README.md` - Website docs
- `website/DEPLOYMENT.md` - Deployment guide
- `website/SETUP.md` - Setup instructions
- `website/PRESENTATION.md` - Presentation script

### 11.3 External Resources
- Next.js Docs: https://nextjs.org/docs
- Tailwind CSS: https://tailwindcss.com/docs
- Mermaid Docs: https://mermaid.js.org/
- React Docs: https://react.dev/

---

## 12. COMPLETION CRITERIA

The website is **COMPLETE** when:

✅ All 20 slides exist and render correctly  
✅ All deep dive buttons work (30+ deep dives implemented)  
✅ All Mermaid diagrams render without errors  
✅ Keyboard navigation works (arrows, space, ESC)  
✅ No TypeScript/build errors  
✅ No console errors in browser  
✅ `npm run build` succeeds  
✅ Static export works offline  
✅ Deployed and accessible via URL  
✅ Presentation takes 20-25 minutes to deliver  
✅ All technical content is accurate  

---

## 13. FINAL CHECKLIST

Before considering this project done:

- [ ] All 30+ deep dives implemented in `deep-dives.tsx`
- [ ] Every "Deep Dive →" button tested and works
- [ ] All 20 slides reviewed for accuracy
- [ ] No fake metrics or demo components remain
- [ ] All Mermaid diagrams render correctly
- [ ] Keyboard navigation fully functional
- [ ] ESC key closes all popups
- [ ] Click-outside closes all popups
- [ ] Progress bar shows correct position
- [ ] Navigation dots highlight current slide
- [ ] Build succeeds: `npm run build`
- [ ] No TypeScript errors: `npx tsc --noEmit`
- [ ] No console errors in browser dev tools
- [ ] Tested on Chrome, Firefox, Safari
- [ ] Tested on mobile, tablet, desktop
- [ ] Deployed to GitHub Pages/Vercel/Netlify
- [ ] Public URL accessible and working
- [ ] README updated with deployment URL
- [ ] All documentation complete

---

**END OF HANDOFF DOCUMENT**

Last Updated: January 7, 2025  
Agent: Kilo Code (Claude Sonnet 4.5)  
Status: Ready for handoff to Codex