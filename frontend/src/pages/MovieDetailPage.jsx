import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { MessageSquare, BarChart2, Activity, Star, Tag, User, Clock, ArrowLeft } from 'lucide-react'
import OracleChat from '../components/OracleChat'
import VibeBar from '../components/VibeBar'
import BingeGauge from '../components/BingeGauge'
import Dock from '../components/ui/Dock'
import { useMovieImages } from '../hooks/useMovieImages'

export default function MovieDetailPage({ movie, onBack }) {
    const [activeTab, setActiveTab] = useState('oracle')
    const images = useMovieImages(movie?.tmdb_id, movie?.title, movie?.year)
    const [activeMediaIndex, setActiveMediaIndex] = useState(0)

    const mediaAssets = useMemo(() => {
        const posters = Array.isArray(images.posters) ? images.posters : (images.poster ? [images.poster] : [])
        const backdrops = Array.isArray(images.backdrops) ? images.backdrops : (images.backdrop ? [images.backdrop] : [])
        const merged = [...posters, ...backdrops].filter(Boolean)
        return [...new Set(merged)].slice(0, 4)
    }, [images.posters, images.backdrops, images.poster, images.backdrop])

    const posterImage = mediaAssets[activeMediaIndex] || images.poster || null

    useEffect(() => {
        setActiveMediaIndex(0)
    }, [movie?.tmdb_id])

    useEffect(() => {
        if (mediaAssets.length <= 1) return

        const id = window.setInterval(() => {
            setActiveMediaIndex((prev) => (prev + 1) % mediaAssets.length)
        }, 5200)

        return () => window.clearInterval(id)
    }, [mediaAssets.length])

    useEffect(() => {
        if (activeMediaIndex >= mediaAssets.length) {
            setActiveMediaIndex(0)
        }
    }, [activeMediaIndex, mediaAssets.length])

    const dockItems = [
        { icon: <MessageSquare size={18} />, label: 'Oracle', onClick: () => setActiveTab('oracle'), active: activeTab === 'oracle' },
        { icon: <BarChart2 size={18} />, label: 'Sentiment', onClick: () => setActiveTab('sentiment'), active: activeTab === 'sentiment' },
        { icon: <Activity size={18} />, label: 'Binge', onClick: () => setActiveTab('binge'), active: activeTab === 'binge' },
    ]

    return (
        <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-4 pb-32">
            {/* Back button */}
            <button
                onClick={onBack}
                className="flex items-center gap-1.5 text-xs text-text-dim hover:text-text-muted
                           mb-4 transition-colors duration-150"
            >
                <ArrowLeft size={13} /> Back to search
            </button>

            {/* Movie header */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-2xl overflow-hidden mb-6 border border-brand-border-subtle bg-brand-surface"
            >
                <div className="p-6 sm:p-8">
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
                {activeTab === 'oracle' && (
                    <div className="grid grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)] gap-6 items-start">
                        <aside className="lg:sticky lg:top-20">
                            <div className="bg-brand-surface border border-brand-border-subtle rounded-2xl overflow-hidden">
                                <div className="bg-brand-card p-3">
                                    {posterImage ? (
                                        <div className="relative w-full aspect-[2/3] rounded-xl overflow-hidden bg-brand-bg">
                                            <AnimatePresence mode="wait" initial={false}>
                                                <motion.img
                                                    key={posterImage}
                                                    src={posterImage}
                                                    alt={movie.title}
                                                    className="absolute inset-0 w-full h-full object-contain"
                                                    loading="lazy"
                                                    decoding="async"
                                                    referrerPolicy="no-referrer"
                                                    initial={{ opacity: 0 }}
                                                    animate={{ opacity: 1 }}
                                                    exit={{ opacity: 0 }}
                                                    transition={{ duration: 0.65, ease: 'easeOut' }}
                                                />
                                            </AnimatePresence>
                                        </div>
                                    ) : (
                                        <div className="w-full aspect-[2/3] rounded-xl bg-gradient-to-br from-brand-bg via-brand-card to-brand-bg p-4 flex flex-col justify-end">
                                            <div className="text-[10px] uppercase tracking-[0.2em] text-brand-gold/70 mb-2">StreamSage</div>
                                            <div className="text-base font-semibold leading-snug text-text-warm line-clamp-4">{movie.title?.toUpperCase() || 'UNTITLED'}</div>
                                            {movie.year && <div className="text-xs text-text-dim mt-2">{movie.year}</div>}
                                        </div>
                                    )}
                                </div>
                                <div className="p-4 border-t border-brand-border-subtle">
                                    {mediaAssets.length > 1 && (
                                        <div className="flex items-center gap-1.5 mb-3">
                                            {mediaAssets.map((_, idx) => (
                                                <button
                                                    key={idx}
                                                    type="button"
                                                    aria-label={`Show poster ${idx + 1}`}
                                                    onClick={() => setActiveMediaIndex(idx)}
                                                    className={`h-1.5 rounded-full transition-all ${idx === activeMediaIndex ? 'w-6 bg-brand-gold' : 'w-3 bg-brand-border-subtle hover:bg-text-dim'}`}
                                                />
                                            ))}
                                        </div>
                                    )}
                                    <p className="text-xs uppercase tracking-[0.16em] text-text-dim mb-2">Oracle Context</p>
                                    <p className="text-sm text-text-muted leading-relaxed">
                                        Ask scene-level questions and get grounded answers with timestamped sources.
                                    </p>
                                </div>
                            </div>
                        </aside>

                        <div className="min-w-0">
                            <OracleChat movieTitle={movie.title} movieRuntime={movie.runtime} />
                        </div>
                    </div>
                )}
                {activeTab === 'sentiment' && <VibeBar movieTitle={movie.title} />}
                {activeTab === 'binge' && <BingeGauge />}
            </motion.div>

            <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
                <Dock items={dockItems} />
            </div>
        </div>
    )
}
