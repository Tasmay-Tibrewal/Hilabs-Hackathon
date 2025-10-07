'use client'

import { useState, useEffect } from 'react'
import DetailSlide from './components/DetailSlide'
import { Slide0, Slide1, Slide2, Slide3, Slide4 } from './slides/slides-0-4'
import { Slide5, Slide7, Slide8 } from './slides/slides-5-8'
import { Slide6 } from './slides/slide-6'
import { Slide9, Slide10, Slide11, Slide12 } from './slides/slides-9-12'
import { Slide13, Slide14, Slide15, Slide16 } from './slides/slides-13-16'
import { Slide17, Slide18, Slide19 } from './slides/slides-17-19'
import { deepDives } from './data/deep-dives'

export default function PresentationPage() {
  const [slide, setSlide] = useState(0)
  const [detail, setDetail] = useState<string | null>(null)

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (detail) { 
        if (e.key === 'Escape') setDetail(null)
        return
      }
      if (e.key === 'ArrowRight' || e.key === ' ') setSlide(s => Math.min(s + 1, 19))
      if (e.key === 'ArrowLeft') setSlide(s => Math.max(s - 1, 0))
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [detail])

  const S = ({ i, children }: { i: number, children: React.ReactNode }) => (
    <section className={`min-h-screen flex items-center justify-center p-12 transition-opacity duration-500 ${
      slide === i ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'
    }`}>{children}</section>
  )

  const showDetail = (id: string) => setDetail(id)

  return (
    <div className="min-h-screen bg-white">
      <div className="fixed top-0 left-0 right-0 h-2 bg-gray-200 z-50">
        <div className="h-full bg-black transition-all" style={{ width: `${((slide + 1) / 20) * 100}%` }} />
      </div>

      <div className="fixed right-6 top-1/2 -translate-y-1/2 z-50 space-y-2">
        {Array.from({ length: 20 }).map((_, i) => (
          <button key={i} onClick={() => setSlide(i)} className={`block w-2 h-2 rounded-full transition-all ${i === slide ? 'bg-black scale-150' : 'bg-gray-300'}`} />
        ))}
      </div>

      <S i={0}><Slide0 /></S>
      <S i={1}><Slide1 onDetail={showDetail} /></S>
      <S i={2}><Slide2 /></S>
      <S i={3}><Slide3 onDetail={showDetail} /></S>
      <S i={4}><Slide4 onDetail={showDetail} /></S>
      <S i={5}><Slide5 onDetail={showDetail} /></S>
      <S i={6}><Slide6 onDetail={showDetail} /></S>
      <S i={7}><Slide7 onDetail={showDetail} /></S>
      <S i={8}><Slide8 onDetail={showDetail} /></S>
      <S i={9}><Slide9 onDetail={showDetail} /></S>
      <S i={10}><Slide10 onDetail={showDetail} /></S>
      <S i={11}><Slide11 onDetail={showDetail} /></S>
      <S i={12}><Slide12 onDetail={showDetail} /></S>
      <S i={13}><Slide13 onDetail={showDetail} /></S>
      <S i={14}><Slide14 onDetail={showDetail} /></S>
      <S i={15}><Slide15 onDetail={showDetail} /></S>
      <S i={16}><Slide16 /></S>
      <S i={17}><Slide17 onDetail={showDetail} /></S>
      <S i={18}><Slide18 /></S>
      <S i={19}><Slide19 /></S>

      {/* Detail Popups - Full Content */}
      {detail && deepDives[detail] && (
        <DetailSlide title={deepDives[detail].title} onClose={() => setDetail(null)}>
          {deepDives[detail].content}
        </DetailSlide>
      )}

      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gray-900 text-white px-6 py-3 rounded-full text-sm font-mono z-40">
        {slide + 1}/20 | ← → or SPACE | Click buttons for details | ESC closes
      </div>
    </div>
  )
}
