'use client'

import { useEffect, useRef, useState } from 'react'

interface MermaidDiagramProps {
  chart: string
  className?: string
}

export default function MermaidDiagram({ chart, className = '' }: MermaidDiagramProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (!isClient || !ref.current) return

    const renderDiagram = async () => {
      try {
        // @ts-ignore - mermaid is loaded via CDN
        if (typeof mermaid !== 'undefined') {
          // @ts-ignore
          mermaid.initialize({ 
            startOnLoad: false, 
            securityLevel: 'loose', 
            theme: 'neutral',
            flowchart: { useMaxWidth: true }
          })
          
          const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`
          // @ts-ignore
          const { svg } = await mermaid.render(id, chart)
          if (ref.current) {
            ref.current.innerHTML = svg
          }
        }
      } catch (error) {
        console.error('Mermaid rendering error:', error)
        if (ref.current) {
          ref.current.innerHTML = '<div class="text-red-600 p-4">Error rendering diagram</div>'
        }
      }
    }

    // Wait a bit for mermaid to load
    const timer = setTimeout(renderDiagram, 100)
    return () => clearTimeout(timer)
  }, [chart, isClient])

  if (!isClient) {
    return (
      <div className={`flex items-center justify-center p-8 bg-gray-50 rounded-xl ${className}`}>
        <div className="text-gray-500">Loading diagram...</div>
      </div>
    )
  }

  return (
    <div 
      ref={ref} 
      className={`mermaid-container flex justify-center ${className}`}
    />
  )
}