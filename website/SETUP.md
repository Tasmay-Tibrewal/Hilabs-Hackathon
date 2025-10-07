# Setup Instructions - Clinical Concept Harmonizer Website

Quick setup guide to get your website running locally and deployed.

## 🎯 Quick Start (5 minutes)

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git (for deployment)

### Local Development

```bash
# Navigate to website directory
cd website

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the site.

## 📦 What's Included

Your website includes all required sections:

### ✅ Implemented Features

1. **Hero Section**
   - Compelling value proposition
   - Three CTA buttons (See Pipeline, Try Demo, Download Repo)
   - Smooth scroll navigation

2. **Problem → Solution**
   - Side-by-side comparison cards
   - Example mappings table with real data
   - Tooltips explaining complexity

3. **Architecture Pipeline**
   - Mermaid flowchart (auto-rendered)
   - 6-step numbered explanation cards
   - Tooltips on each step

4. **Interactive Scoring Demo**
   - 4 adjustable weight sliders
   - Live formula display
   - Cosine normalization explanation

5. **Live Demo Component**
   - Text input + entity type dropdown
   - Sample data simulation
   - Results with "Why this?" expandable details
   - Anchor and fuzzy matching explanation

6. **Performance & Configuration**
   - 6 configuration cards with tooltips
   - Copy-paste command examples
   - Build, batch, and query examples

7. **Future Work**
   - 12 enhancement cards
   - Categories: indexing, LLM, lexical, clustering, etc.

8. **Credits & Technology**
   - Core libraries
   - LLM backends
   - Vocabularies
   - Licensing information

9. **Professional Footer**
   - Three-column layout
   - Links to resources
   - Deploy instructions note

### 🎨 Design Features

- **Monochrome Theme**: Black/white/gray palette (OpenAI/Uber style)
- **Responsive**: Mobile, tablet, desktop optimized
- **Tooltips**: Hover `?` icons for detailed explanations
- **Progressive Disclosure**: Short by default, expandable for details
- **Smooth Animations**: Fade-ins, hover effects
- **Professional Typography**: Clean, readable fonts

## 🔧 Customization

### 1. Update Repository Links

Find and replace in `app/page.tsx`:
```tsx
// Current (line ~68, ~74, ~103)
href="https://github.com/yourusername/clinical-harmonizer"

// Replace with your actual GitHub URL
href="https://github.com/YOUR_USERNAME/YOUR_REPO"
```

### 2. Modify Sample Data

Edit the demo data in `app/page.tsx` (line ~1023):
```tsx
const sampleData = {
  'Your Input': {
    candidates: [
      { system: 'RXNORM', code: '123', ... }
    ]
  }
}
```

### 3. Adjust Scoring Weights

Default weights in `app/page.tsx` (line ~6):
```tsx
const [weights, setWeights] = useState({
  desc: 0.30,    // Description match
  kw: 0.40,      // Keyword match
  direct: 0.20,  // Direct query match
  sty: 0.10      // Semantic type match
})
```

### 4. Add More Mermaid Diagrams

In `app/page.tsx`, add new diagram sections:
```tsx
<pre className="mermaid">
{`flowchart LR
  A[Start] --> B[Process]
  B --> C[End]`}
</pre>
```

### 5. Customize Colors

While maintaining monochrome, adjust shades in `tailwind.config.ts`:
```typescript
colors: {
  gray: {
    // Customize these values
    100: '#f3f4f6',
    // ... etc
  }
}
```

## 📱 Testing Responsive Design

### Desktop
```bash
npm run dev
# Visit http://localhost:3000
```

### Mobile Simulation
1. Open Chrome DevTools (F12)
2. Click device toolbar icon (Ctrl+Shift+M)
3. Select device (iPhone, iPad, etc.)

### Real Device Testing
1. Find your local IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Start dev server: `npm run dev`
3. On mobile device, visit: `http://YOUR_IP:3000`

## 🚀 Production Build

```bash
# Build static site
npm run build

# Preview production build locally
npx serve out

# Visit http://localhost:3000 to verify
```

The `out` directory contains your complete static website.

## 📂 Project Structure

```
website/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions deployment
├── app/
│   ├── layout.tsx              # Root layout, Mermaid setup
│   ├── page.tsx                # Main page (all sections)
│   └── globals.css             # Tailwind + custom styles
├── public/                     # Static assets (future)
├── .gitignore                  # Git ignore rules
├── next.config.js              # Next.js static export config
├── package.json                # Dependencies
├── postcss.config.js           # PostCSS for Tailwind
├── tailwind.config.ts          # Tailwind configuration
├── tsconfig.json               # TypeScript config
├── README.md                   # Main documentation
├── DEPLOYMENT.md               # Detailed deploy guide
└── SETUP.md                    # This file
```

## 🔍 Key Files Explained

### `app/page.tsx` (Main Content)
- All sections in one file for easy editing
- Interactive components (Demo, Sliders)
- Mermaid diagrams embedded as strings
- Sample data for demo

### `app/globals.css` (Styling)
- Tailwind directives
- Custom tooltip styles
- Mermaid diagram centering
- Scroll behavior

### `next.config.js` (Configuration)
- `output: 'export'` for static generation
- `images: { unoptimized: true }` for static export
- `trailingSlash: true` for GitHub Pages compatibility

## 🐛 Common Issues & Fixes

### Issue: TypeScript errors in editor
**Fix**: Run `npm install` to install all dependencies including TypeScript types.

### Issue: Styles not applying
**Fix**: 
```bash
rm -rf .next
npm run dev
```

### Issue: Mermaid diagrams not rendering
**Fix**: 
1. Check browser console for CDN errors
2. Verify diagram syntax (no extra spaces, correct arrows)
3. Refresh page after code changes

### Issue: Port 3000 already in use
**Fix**: 
```bash
# Kill process on port 3000
# Windows: netstat -ano | findstr :3000, then taskkill /PID <PID>
# Mac/Linux: lsof -ti:3000 | xargs kill

# Or use different port
npm run dev -- -p 3001
```

### Issue: Build fails
**Fix**:
```bash
rm -rf node_modules .next out
npm install
npm run build
```

## 📊 Performance Checklist

After building, verify:
- [ ] Build completes without errors
- [ ] All pages in `out` directory
- [ ] Total size < 5MB
- [ ] Images optimized
- [ ] No console errors when loading
- [ ] Lighthouse score > 90

## 🌐 Browser Compatibility

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile Safari (iOS 14+)
- ✅ Chrome Mobile (Android 9+)

## 📈 Next Steps

1. **Test Locally**: Run `npm run dev` and verify everything works
2. **Customize**: Update GitHub URLs and sample data
3. **Build**: Run `npm run build` to generate static site
4. **Deploy**: Follow `DEPLOYMENT.md` for deployment options
5. **Monitor**: Check deployment status in GitHub Actions

## 💡 Tips

- **Hot Reload**: Changes auto-refresh in dev mode
- **Component Isolation**: Test components individually
- **Mobile First**: Test mobile view while developing
- **Use Tooltips**: Add `?` tooltips for complex concepts
- **Keep It Fast**: Static site = instant loads

## 📚 Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Mermaid Syntax](https://mermaid.js.org/syntax/flowchart.html)
- [GitHub Pages Guide](https://docs.github.com/en/pages)

## 🆘 Getting Help

1. Check `DEPLOYMENT.md` for deployment issues
2. Review `README.md` for general information
3. Search Next.js/Tailwind documentation
4. Check browser console for errors
5. Open GitHub issue with error details

---

**Ready to Deploy?** See `DEPLOYMENT.md` for step-by-step instructions.

**Need to Customize?** All content is in `app/page.tsx` - easy to find and edit.

**Questions?** Check the FAQ section in `README.md`.