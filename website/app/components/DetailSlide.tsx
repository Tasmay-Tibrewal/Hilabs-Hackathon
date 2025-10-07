'use client'

interface DetailSlideProps {
  title: string
  children: React.ReactNode
  onClose: () => void
}

export default function DetailSlide({ title, children, onClose }: DetailSlideProps) {
  return (
    <div 
      className="fixed inset-0 bg-black/90 flex items-center justify-center z-[100] p-8 overflow-auto"
      onClick={onClose}
    >
      <div 
        className="bg-white rounded-3xl max-w-7xl w-full max-h-[90vh] overflow-auto p-12 relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-6 right-6 w-12 h-12 flex items-center justify-center text-4xl hover:bg-gray-100 rounded-full transition-colors"
        >
          ×
        </button>
        <h3 className="text-5xl font-bold mb-8 pr-16">{title}</h3>
        <div className="text-xl leading-relaxed">
          {children}
        </div>
        <div className="mt-12 text-center text-gray-500 text-lg">
          Press ESC or click outside to close
        </div>
      </div>
    </div>
  )
}