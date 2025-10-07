# Clinical Concept Harmonizer - Website

Professional, interactive website showcasing the Clinical Concept Harmonizer solution for the HiLabs Hackathon 2025.

## 🚀 Quick Start

### Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
# Build static site
npm run build

# The static site will be in the 'out' directory
```

## 📦 Deploy to GitHub Pages

### Option 1: GitHub Actions (Recommended)

1. Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: cd website && npm ci
        
      - name: Build
        run: cd website && npm run build
        
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./website/out

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

2. In your GitHub repository:
   - Go to **Settings** → **Pages**
   - Under "Source", select **GitHub Actions**
   - Push to main branch to trigger deployment

### Option 2: Manual Deployment

```bash
# Build the site
npm run build

# The 'out' directory contains your static site
# Copy contents to your hosting provider or GitHub Pages
```

For GitHub Pages manual deployment:
1. Copy contents of `out` folder to your `gh-pages` branch
2. Enable GitHub Pages in repository settings
3. Select `gh-pages` branch as source

## 🎨 Features

- **Responsive Design**: Works on all devices (mobile, tablet, desktop)
- **Interactive Demo**: Client-side simulation of the matching process
- **Mermaid Diagrams**: Auto-rendered architecture flowcharts
- **Tooltips**: Hover hints on complex concepts
- **Monochrome Theme**: Professional black/white design
- **Fast & Lightweight**: Static site with no backend dependencies

## 📁 Project Structure

```
website/
├── app/
│   ├── layout.tsx       # Root layout with Mermaid.js
│   ├── page.tsx         # Main page with all sections
│   └── globals.css      # Global styles + Tailwind
├── public/              # Static assets (if any)
├── package.json         # Dependencies
├── next.config.js       # Next.js config (static export)
├── tailwind.config.ts   # Tailwind CSS config
└── tsconfig.json        # TypeScript config
```

## 🛠️ Technology Stack

- **Framework**: Next.js 14 (Static Export)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Diagrams**: Mermaid.js (CDN)
- **Hosting**: GitHub Pages / Vercel / Netlify

## 🌐 Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## 📝 Customization

### Update GitHub Repository Link

In `app/page.tsx`, replace:
```tsx
href="https://github.com/yourusername/clinical-harmonizer"
```

with your actual repository URL.

### Modify Mermaid Diagrams

The main architecture diagram is embedded in `app/page.tsx`. You can:
1. Edit the mermaid code directly in the page
2. Add new diagrams using the same `<pre className="mermaid">` format

### Adjust Weights & Scoring

The interactive scoring sliders use default weights:
- Description: 0.30
- Keywords: 0.40
- Direct: 0.20
- STY: 0.10

Modify these in the `weights` state in `app/page.tsx`.

## 🔧 Troubleshooting

### Build Fails

```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
npm run build
```

### Mermaid Not Rendering

- Ensure CDN is accessible
- Check browser console for errors
- Verify diagram syntax in mermaid code blocks

### Styles Not Applied

```bash
# Rebuild Tailwind
npm run dev
```

## 📄 License

This website is part of the Clinical Concept Harmonizer project.

## 🙋 Support

For issues or questions about the website:
1. Check the main repository README
2. Open an issue on GitHub
3. Contact the team

---

Built with ❤️ for HiLabs Hackathon 2025