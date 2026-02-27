import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Clapperboard, ChevronLeft, Cpu, Wifi } from 'lucide-react'
import { getModelInfo } from '../api'

export default function Header({ currentPage, onNavigateHome }) {
    const [modelInfo, setModelInfo] = useState(null)
    const [scrolled, setScrolled] = useState(false)

    useEffect(() => {
        getModelInfo().then(data => setModelInfo(data)).catch(() => {})
        const onScroll = () => setScrolled(window.scrollY > 10)
        window.addEventListener('scroll', onScroll, { passive: true })
        return () => window.removeEventListener('scroll', onScroll)
    }, [])

    const getModelLabel = () => {
        if (!modelInfo) return null
        const backend = modelInfo.inference_backend || ''
        const model = modelInfo.llm_model || ''
        const shortName = model.includes('\\') || model.includes('/') ? model.split(/[/\\]/).pop() : model
        if (shortName.includes('final_model') || backend === 'local')
            return { name: 'Llama 3 8B', type: 'local' }
        return { name: shortName || backend, type: backend }
    }

    const label = getModelLabel()
    const isHealthy = modelInfo?.status === 'healthy'

    return (
        <header className={`sticky top-0 z-50 transition-all duration-300 ${
            scrolled
                ? 'bg-brand-bg/90 backdrop-blur-md border-b border-brand-border-subtle'
                : 'bg-transparent'
        }`}>
            <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
                <motion.button
                    onClick={onNavigateHome}
                    className="flex items-center gap-2.5 group"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                >
                    <div className="w-7 h-7 bg-brand-gold rounded-lg flex items-center justify-center
                                    group-hover:shadow-gold transition-shadow duration-200">
                        <Clapperboard size={14} className="text-brand-bg" />
                    </div>
                    <span className="text-text-warm font-bold text-base tracking-tight hidden sm:inline">
                        StreamSage
                    </span>
                </motion.button>

                <div className="flex items-center gap-3">
                    {currentPage === 'detail' && (
                        <button onClick={onNavigateHome} className="btn-ghost text-sm">
                            <ChevronLeft size={15} /> Back
                        </button>
                    )}
                    {label && (
                        <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-lg
                                        bg-brand-surface border border-brand-border-subtle text-xs">
                            <div className={`w-1.5 h-1.5 rounded-full ${isHealthy ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                            <span className="text-text-muted font-medium">{label.name}</span>
                            <span className={`flex items-center gap-0.5 text-[10px] font-bold px-1.5 py-0.5 rounded ${
                                label.type === 'local' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-blue-500/10 text-blue-400'
                            }`}>
                                {label.type === 'local' ? <Cpu size={8} /> : <Wifi size={8} />}
                                {label.type === 'local' ? 'LOCAL' : 'API'}
                            </span>
                        </div>
                    )}
                </div>
            </div>
        </header>
    )
}
