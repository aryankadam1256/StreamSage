import { useState } from 'react'
import { motion } from 'framer-motion'
import { MessageSquare, BarChart2, Activity, Star, Tag, User, Clock, ArrowLeft } from 'lucide-react'
import OracleChat from '../components/OracleChat'
import VibeBar from '../components/VibeBar'
import BingeGauge from '../components/BingeGauge'
import Dock from '../components/ui/Dock'
import LetterGlitch from '../components/ui/LetterGlitch'

export default function MovieDetailPage({ movie, onBack }) {
    const [activeTab, setActiveTab] = useState('oracle')

    const dockItems = [
        { icon: <MessageSquare size={18} />, label: 'Oracle', onClick: () => setActiveTab('oracle'), active: activeTab === 'oracle' },
        { icon: <BarChart2 size={18} />, label: 'Sentiment', onClick: () => setActiveTab('sentiment'), active: activeTab === 'sentiment' },
        { icon: <Activity size={18} />, label: 'Binge', onClick: () => setActiveTab('binge'), active: activeTab === 'binge' },
    ]

    return (
        <div className="max-w-5xl mx-auto px-4 sm:px-6 pt-4 pb-32">
            {/* Back button */}
            <button
                onClick={onBack}
                className="flex items-center gap-1.5 text-xs text-text-dim hover:text-text-muted
                           mb-4 transition-colors duration-150"
            >
                <ArrowLeft size={13} /> Back to search
            </button>

            {/* Movie header with LetterGlitch background */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative rounded-2xl overflow-hidden mb-8 border border-brand-border-subtle"
            >
                <div className="absolute inset-0 opacity-20">
                    <LetterGlitch
                        glitchColors={['#d4a017', '#a07c12', '#1a1a2e']}
                        glitchSpeed={80}
                        outerVignette={true}
                        centerVignette={true}
                        smooth={true}
                        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    />
                </div>
                <div className="relative z-10 p-6 sm:p-8 bg-gradient-to-r from-brand-bg/90 via-brand-bg/80 to-transparent">
                    <h1 className="text-2xl sm:text-3xl font-bold text-text-warm mb-1">
                        {movie.title}
                        {movie.year && <span className="text-text-dim font-normal text-lg ml-3">({movie.year})</span>}
                    </h1>
                    <div className="flex gap-2 mt-3 flex-wrap">
                        {(movie.genres || movie.genre) && (
                            <span className="flex items-center gap-1.5 text-xs text-text-muted bg-brand-card/60
                                             px-3 py-1.5 rounded-lg border border-brand-border-subtle">
                                <Tag size={11} /> {movie.genres || movie.genre}
                            </span>
                        )}
                        {movie.director && (
                            <span className="flex items-center gap-1.5 text-xs text-text-muted bg-brand-card/60
                                             px-3 py-1.5 rounded-lg border border-brand-border-subtle">
                                <User size={11} /> {movie.director}
                            </span>
                        )}
                        {movie.rating && (
                            <span className="flex items-center gap-1.5 text-xs text-amber-400 bg-amber-400/10
                                             px-3 py-1.5 rounded-lg border border-amber-400/20">
                                <Star size={11} className="fill-amber-400" /> {movie.rating}
                            </span>
                        )}
                        {movie.runtime && (
                            <span className="flex items-center gap-1.5 text-xs text-text-muted bg-brand-card/60
                                             px-3 py-1.5 rounded-lg border border-brand-border-subtle">
                                <Clock size={11} />
                                {Math.floor(movie.runtime / 60)}h {movie.runtime % 60}m
                            </span>
                        )}
                    </div>
                    {movie.description && (
                        <p className="mt-4 text-text-muted text-sm leading-relaxed max-w-xl">{movie.description}</p>
                    )}
                </div>
            </motion.div>

            <motion.div key={activeTab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}>
                {activeTab === 'oracle' && <OracleChat movieTitle={movie.title} movieRuntime={movie.runtime} />}
                {activeTab === 'sentiment' && <VibeBar movieTitle={movie.title} />}
                {activeTab === 'binge' && <BingeGauge />}
            </motion.div>

            <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
                <Dock items={dockItems} />
            </div>
        </div>
    )
}
