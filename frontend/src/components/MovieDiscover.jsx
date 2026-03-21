/**
 * =============================================================================
 * Movie Discovery Component - Fine-tuned Llama 3 Assistant
 * =============================================================================
 *
 * Uses:
 * - Fine-tuned Llama 3 8B (QLoRA, trained on 2946 movie Q&A pairs)
 * - RAG: ChromaDB semantic search augments the LLM with real movie data
 * - Per-movie recommendation reasons explaining WHY each movie matches
 *
 * =============================================================================
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

const EXAMPLE_QUERIES = [
    "Recommend mind-bending sci-fi movies like Inception",
    "What are some great horror movies under 2 hours?",
    "I want something funny with a happy ending",
    "Show me Christopher Nolan films",
    "Best animated movies for adults",
    "Movies similar to The Dark Knight",
    "I'm stressed and want to relax with a feel-good movie",
]

export default function MovieDiscover() {
    const [query, setQuery] = useState('')
    const [genre, setGenre] = useState('')
    const [minRating, setMinRating] = useState('')
    const [response, setResponse] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleDiscover = async (e) => {
        e.preventDefault()
        if (!query.trim()) return

        setLoading(true)
        setError(null)
        setResponse(null)

        try {
            const payload = {
                query: query.trim(),
                filters: {
                    ...(genre && { genre }),
                    ...(minRating && { min_rating: parseFloat(minRating) }),
                },
                top_k: 5,
            }

            const res = await axios.post(`${API_URL}/discover`, payload)
            setResponse(res.data)
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to get recommendations')
        } finally {
            setLoading(false)
        }
    }

    const handleExample = (example) => {
        setQuery(example)
    }

    // Use recommended_movies from API (structured data)
    const movies = response?.recommended_movies || []

    return (
        <div className="card-cyber max-w-4xl mx-auto">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="text-4xl">🎬</div>
                <div>
                    <h2 className="text-2xl font-bold text-cyber-accent">Movie Discovery</h2>
                    <p className="text-gray-400">
                        Fine-tuned Llama 3 8B · RAG · {' '}
                        <span className="text-cyber-purple text-xs">QLoRA · Semantic Search</span>
                    </p>
                </div>
            </div>

            {/* Example queries */}
            <div className="mb-5">
                <p className="text-xs text-gray-400 mb-2">Try an example:</p>
                <div className="flex flex-wrap gap-2">
                    {EXAMPLE_QUERIES.map((q, i) => (
                        <button
                            key={i}
                            onClick={() => handleExample(q)}
                            className="px-3 py-1 text-xs rounded-full glass-morphism text-gray-300
                                       hover:bg-white/10 hover:text-cyber-accent transition-all"
                        >
                            {q}
                        </button>
                    ))}
                </div>
            </div>

            {/* Query Form */}
            <form onSubmit={handleDiscover} className="space-y-4 mb-6">
                <div>
                    <label className="block text-sm font-medium mb-2 text-gray-300">
                        What are you looking for?
                    </label>
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        rows={3}
                        className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-accent/30
                                   focus:border-cyber-accent focus:outline-none transition-colors resize-none"
                        placeholder="e.g. Recommend mind-bending sci-fi movies like Inception..."
                    />
                </div>

                {/* Optional Filters */}
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium mb-2 text-gray-300">
                            Genre (optional)
                        </label>
                        <input
                            type="text"
                            value={genre}
                            onChange={(e) => setGenre(e.target.value)}
                            className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-accent/30
                                       focus:border-cyber-accent focus:outline-none transition-colors"
                            placeholder="e.g. Action, Horror, Comedy"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium mb-2 text-gray-300">
                            Min Rating (optional)
                        </label>
                        <input
                            type="number"
                            step="0.1"
                            min="0"
                            max="10"
                            value={minRating}
                            onChange={(e) => setMinRating(e.target.value)}
                            className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-accent/30
                                       focus:border-cyber-accent focus:outline-none transition-colors"
                            placeholder="e.g. 7.5"
                        />
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="btn-cyber w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? (
                        <span className="flex items-center justify-center gap-2">
                            <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                            Searching movies...
                        </span>
                    ) : (
                        '🎬 Discover Movies'
                    )}
                </button>
            </form>

            {/* Results */}
            <AnimatePresence mode="wait">
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400"
                    >
                        <p className="font-semibold">Error</p>
                        <p className="text-sm">{error}</p>
                    </motion.div>
                )}

                {response && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="space-y-4"
                    >
                        {/* Performance metrics & model info */}
                        <div className="flex flex-wrap gap-4 text-xs text-gray-400 mb-2">
                            {response.retrieval_count !== undefined && (
                                <span>🎞 {response.retrieval_count} movies found</span>
                            )}
                            {response.model_used && (
                                <span className="text-cyber-purple">🤖 {response.model_used.includes('local') ? 'Fine-tuned Llama 3' : response.model_used}</span>
                            )}
                        </div>

                        {/* Movie Cards with per-movie explanations */}
                        {movies.length > 0 ? (
                            <div className="space-y-4">
                                {movies.map((movie, idx) => {
                                    const isEvenRow = idx % 2 === 0;
                                    return (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: idx * 0.1 }}
                                            className="p-4 rounded-lg glass-morphism border border-cyber-accent/20
                                                       hover:border-cyber-accent/50 transition-all flex flex-col md:flex-row gap-6"
                                        >
                                            <div className={`flex flex-col flex-1 ${!isEvenRow ? 'md:order-2' : ''}`}>
                                                {/* Movie Header */}
                                                <div className="flex items-start justify-between gap-2 mb-2">
                                                    <h3 className="font-bold text-white text-lg">
                                                        <span className="text-cyber-accent mr-2">#{idx + 1}</span>
                                                        {movie.title}
                                                        {movie.year && (
                                                            <span className="text-gray-400 font-normal ml-2">({movie.year})</span>
                                                        )}
                                                    </h3>
                                                    <div className="flex gap-2 text-xs shrink-0">
                                                        {movie.rating && (
                                                            <span className="px-2 py-1 rounded-full bg-yellow-500/20 text-yellow-400 font-medium">
                                                                ⭐ {movie.rating.toFixed(1)}
                                                            </span>
                                                        )}
                                                        {movie.relevance_score && (
                                                            <span className="px-2 py-1 rounded-full bg-green-500/20 text-green-400 font-medium">
                                                                {(movie.relevance_score * 100).toFixed(0)}% match
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>

                                                {/* Movie Meta */}
                                                <div className="flex flex-wrap gap-2 mb-3">
                                                    {movie.genres && (
                                                        <span className="text-xs px-2 py-1 rounded-full bg-cyber-purple/20 text-cyber-purple">
                                                            {movie.genres}
                                                        </span>
                                                    )}
                                                    {movie.director && (
                                                        <span className="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-400">
                                                            🎬 {movie.director}
                                                        </span>
                                                    )}
                                                    {movie.runtime && (
                                                        <span className="text-xs px-2 py-1 rounded-full bg-gray-500/20 text-gray-400">
                                                            ⏱ {movie.runtime} min
                                                        </span>
                                                    )}
                                                </div>

                                                {/* Movie Description */}
                                                {movie.description && (
                                                    <p className="text-gray-400 text-sm leading-relaxed">
                                                        {movie.description}
                                                    </p>
                                                )}
                                            </div>

                                            {/* AI Explanation Content - alternating sides */}
                                            {movie.recommendation_reason && (
                                                <div className={`flex flex-col w-full md:w-1/3 shrink-0 ${!isEvenRow ? 'md:order-1' : ''}`}>
                                                    <div className="h-full p-4 rounded-lg bg-gradient-to-br from-cyber-purple/10 to-cyber-pink/10
                                                                    border-l-4 border-cyber-accent shadow-inner flex flex-col justify-center gap-2">
                                                        <h4 className="flex items-center gap-2 text-sm font-bold text-cyber-accent uppercase tracking-wider">
                                                            <span><svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg></span>
                                                            AI's Take
                                                        </h4>
                                                        <p className="text-sm text-gray-200 leading-relaxed italic">
                                                            "{movie.recommendation_reason}"
                                                        </p>
                                                    </div>
                                                </div>
                                            )}
                                        </motion.div>
                                    );
                                })}
                            </div>
                        ) : (
                            /* No movies found */
                            <div className="p-4 rounded-lg glass-morphism text-center">
                                <p className="text-gray-400">No movies found matching your criteria.</p>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Help */}
            {!response && !error && !loading && (
                <div className="mt-2 p-4 rounded-lg glass-morphism">
                    <p className="text-sm text-gray-400">
                        💡 <strong>How it works:</strong> Your query is converted to a vector embedding,
                        ChromaDB retrieves semantically similar movies, and our fine-tuned{' '}
                        <span className="text-cyber-accent">Llama 3</span> explains why each movie
                        matches your search. Results are sorted by <strong>relevance</strong>, not rating.
                    </p>
                </div>
            )}
        </div>
    )
}
