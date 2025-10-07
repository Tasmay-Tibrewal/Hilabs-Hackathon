# Deployment Guide - Clinical Concept Harmonizer Website

This guide covers multiple deployment options for the static Next.js website.

## 📋 Pre-Deployment Checklist

- [ ] Update GitHub repository URL in `app/page.tsx` (search for "yourusername")
- [ ] Test build locally: `npm run build`
- [ ] Verify all pages render correctly
- [ ] Check mobile responsiveness
- [ ] Test Mermaid diagrams load properly
- [ ] Ensure all tooltips work

## 🚀 Deployment Options

### Option 1: GitHub Pages (Recommended)

#### Automatic Deployment via GitHub Actions

1. **Enable GitHub Pages**:
   - Go to your repository **Settings** → **Pages**
   - Under "Build and deployment" → "Source", select **GitHub Actions**

2. **Push to Repository**:
   ```bash
   git add .
   git commit -m "Add website"
   git push origin main
   ```

3. **Monitor Deployment**:
   - Go to **Actions** tab in your repository
   - Watch the "Deploy to GitHub Pages" workflow
   - Once complete, your site will be live at: `https://yourusername.github.io/repo-name/`

#### Manual GitHub Pages Deployment

```bash
# Build the site
cd website
npm install
npm run build

# Deploy to gh-pages branch
npx gh-pages -d out
```

### Option 2: Vercel

1. **Install Vercel CLI** (optional):
   ```bash
   npm i -g vercel
   ```

2. **Deploy**:
   ```bash
   cd website
   vercel
   ```

3. **Or use Vercel Dashboard**:
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set Root Directory to `website`
   - Deploy

### Option 3: Netlify

1. **Netlify CLI**:
   ```bash
   npm install -g netlify-cli
   cd website
   npm run build
   netlify deploy --prod --dir=out
   ```

2. **Or use Netlify Dashboard**:
   - Go to [netlify.com](https://netlify.com)
   - New site from Git
   - Select your repository
   - Build command: `cd website && npm run build`
   - Publish directory: `website/out`

### Option 4: Custom Server

```bash
# Build
cd website
npm run build

# The 'out' folder contains your static site
# Copy to your web server:
scp -r out/* user@yourserver.com:/var/www/html/
```

## 🔧 Configuration

### Base Path (for subdirectory deployment)

If deploying to a subdirectory (e.g., `yourdomain.com/clinical-harmonizer/`):

1. Edit `next.config.js`:
```javascript
const nextConfig = {
  output: 'export',
  basePath: '/clinical-harmonizer',  // Add this
  images: { unoptimized: true },
  trailingSlash: true,
}
```

2. Update all internal links to use `basePath`.

### Custom Domain

#### GitHub Pages:
1. Create `public/CNAME` file:
   ```
   yourdomain.com
   ```
2. In GitHub Settings → Pages, add custom domain
3. Configure DNS:
   - For apex domain: A records to GitHub IPs
   - For subdomain: CNAME to `yourusername.github.io`

#### Vercel/Netlify:
- Follow their dashboard instructions for custom domains

## 🐛 Troubleshooting

### Build Errors

```bash
# Clear cache and rebuild
cd website
rm -rf node_modules .next out
npm install
npm run build
```

### 404 on Routes

- Ensure `trailingSlash: true` in `next.config.js`
- For SPA routing, configure server to serve `index.html` for all routes

### Mermaid Diagrams Not Rendering

- Check browser console for CDN loading errors
- Verify diagram syntax
- Try different CDN: `https://unpkg.com/mermaid@10/dist/mermaid.min.js`

### Styles Not Loading

```bash
# Rebuild Tailwind
cd website
npm run dev  # Check if styles work locally first
npm run build
```

### Images Not Loading

- Place images in `public/` folder
- Reference as `/image.png` (not `public/image.png`)
- Ensure `images: { unoptimized: true }` in config

## 📊 Performance Optimization

### Already Included:
- ✅ Static export (no server needed)
- ✅ Tailwind CSS (purged unused styles)
- ✅ Next.js optimization
- ✅ CDN for external libraries

### Additional Options:

1. **Minify HTML** (add to package.json):
   ```json
   "scripts": {
     "postbuild": "cd out && find . -name '*.html' -exec html-minifier --collapse-whitespace {} \\;"
   }
   ```

2. **Compress Assets**:
   ```bash
   # After build
   cd out
   gzip -r .
   ```

3. **CDN Deployment**:
   - Upload `out` folder to CloudFlare Pages
   - Or use AWS S3 + CloudFront

## 🔒 Security Headers

For production, add security headers:

### Netlify (`netlify.toml`):
```toml
[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
```

### Vercel (`vercel.json`):
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Frame-Options", "value": "DENY" },
        { "key": "X-Content-Type-Options", "value": "nosniff" }
      ]
    }
  ]
}
```

## 📈 Analytics (Optional)

Add to `app/layout.tsx` before `</head>`:

### Google Analytics:
```tsx
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script dangerouslySetInnerHTML={{
  __html: `
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-XXXXXXXXXX');
  `
}} />
```

### Plausible (Privacy-friendly):
```tsx
<script defer data-domain="yourdomain.com" src="https://plausible.io/js/script.js"></script>
```

## 🧪 Testing Deployment Locally

Serve the built site locally to test:

```bash
# Build
npm run build

# Serve with a static server
npx serve out

# Or use Python
cd out && python -m http.server 8000

# Or use Node
npx http-server out -p 8000
```

Visit `http://localhost:8000` to test.

## 📱 Progressive Web App (Optional)

To make it a PWA, add `public/manifest.json` and service worker.

## ✅ Post-Deployment Checklist

- [ ] Visit deployed URL
- [ ] Test on mobile device
- [ ] Check all sections load
- [ ] Verify Mermaid diagrams render
- [ ] Test interactive demo
- [ ] Check tooltips on hover
- [ ] Test navigation links
- [ ] Verify GitHub links work
- [ ] Check page load speed (< 3s)
- [ ] Test on different browsers

## 🔄 Updating the Site

```bash
# Make changes to code
# Test locally
npm run dev

# Build and deploy
npm run build
git add .
git commit -m "Update website"
git push origin main  # GitHub Actions will auto-deploy
```

## 📞 Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review Next.js static export docs
3. Check GitHub Actions logs (if using GH Pages)
4. Open an issue in the repository

---

**Last Updated**: 2025-01-06