import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Clinical Concept Harmonizer | HiLabs Hackathon 2025',
  description: 'Normalize messy clinical inputs to standardized RxNorm and SNOMED CT codes—fast, robust, and explainable.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}