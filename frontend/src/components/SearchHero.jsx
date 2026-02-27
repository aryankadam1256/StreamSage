import { useState } from 'react'
import { motion } from 'framer-motion'
import { Search, SlidersHorizontal, Loader2 } from 'lucide-react'
import BlurText from './ui/BlurText'
import Particles from './ui/Particles'

const EXAMPLES = [
    "Something dark and mind-bending like Inception",
    "Uplifting movies about chasing your dreams",
    "Best horror films with psychological twists",
    "Feel-good comedies for a rainy evening",
]

export default function SearchHero({ onSearch, loading }) {
    const [query, setQuery] = useState('')
    const [showFilters, setShowFilters] = useState(false)
    const [genre, setGenre] = useState('')
    const [minRating, setMinRating] = useState('')

    const handleSubmit = (e) => {
        e.preventDefault()
        if (!query.trim()) return
        onSearch(query.trim(), {
            ...(genre && { genre }),
            ...(minRating && { min_rating: parseFloat(minRating) }),
        })
    }

    return (
        <section className="relative min-h-[70vh] flex flex-col items-center justify-center px-4 py-20 overflow-hidden">
            {/* Particles background */}
            <Particles count={600} color={[0.83, 0.63, 0.09]} spread={1.5} speed={0.2} alpha={0.35} />
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-brand-bg/60 to-brand-bg pointer-events-none z-[1]" />

            <div className="relative z-10 w-full max-w-2xl mx-auto">
                <div className="text-center mb-10">
                    <h1 className="text-5xl sm:text-6xl md:text-7xl font-black tracking-tighter leading-none mb-3">
                        <BlurText text="StreamSage" delay={0.1} className="text-text-warm" />
                    </h1>
                    <motion.p
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.8, duration: 0.5 }}
                        className="text-text-muted text-sm sm:text-base"
                    >
                        AI-powered film discovery. Tell us what you're in the mood for.
                    </motion.p>
                </div>

                <motion.form
                    onSubmit={handleSubmit}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 1, duration: 0.5 }}
                    className="space-y-3"
                >
                    <div className="flex gap-2">
                        <div className="relative flex-1">
                            <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-text-dim" />
                            <input
                                type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                                placeholder="Describe a movie you'd love to watch..."
                                className="w-full bg-brand-surface border border-brand-border rounded-xl pl-11 pr-4 py-4
                                           text-text-warm placeholder:text-text-dim text-sm
                                           focus:outline-none focus:border-brand-gold/40 focus:shadow-gold
                                           transition-all duration-200"
                            />
                        </div>
                        <button type="submit" disabled={loading || !query.trim()}
                            className="btn-primary rounded-xl px-5 disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none">
                            {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
                        </button>
                        <button type="button" onClick={() => setShowFilters(v => !v)}
                            className={`btn-secondary rounded-xl px-3 ${showFilters ? 'border-brand-gold/30 text-brand-gold' : ''}`}>
                            <SlidersHorizontal size={15} />
                        </button>
                    </div>

                    {showFilters && (
                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                            className="grid grid-cols-2 gap-3 overflow-hidden">
                            <div>
                                <label className="block text-xs text-text-muted mb-1">Genre</label>
                                <input type="text" value={genre} onChange={(e) => setGenre(e.target.value)}
                                    placeholder='e.g. "Sci-Fi"' className="input-field text-sm py-2" />
                            </div>
                            <div>
                                <label className="block text-xs text-text-muted mb-1">Min rating</label>
                                <input type="number" min="0" max="10" step="0.1"
                                    value={minRating} onChange={(e) => setMinRating(e.target.value)}
                                    placeholder="7.0" className="input-field text-sm py-2" />
                            </div>
                        </motion.div>
                    )}
                </motion.form>

                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    transition={{ delay: 1.3, duration: 0.6 }}
                    className="flex flex-wrap justify-center gap-2 mt-6">
                    {EXAMPLES.map((ex, i) => (
                        <button key={i} onClick={() => { setQuery(ex); onSearch(ex, {}) }}
                            className="px-3 py-1.5 text-xs text-text-dim rounded-full border border-brand-border-subtle
                                       hover:border-brand-gold/20 hover:text-text-muted transition-all duration-200">
                            {ex}
                        </button>
                    ))}
                </motion.div>
            </div>
        </section>
    )
}
