/**
 * =============================================================================
 * Movie Discovery Component - Fine-tuned Llama 3 Assistant
 * =============================================================================
 *
 * 🎓 CONCEPT: Conversational AI with RAG
 *
 * This component interfaces with the Movie Assistant Service which uses:
 * - Fine-tuned Llama 3 8B (QLoRA, trained on 2946 movie Q&A pairs)
 * - RAG: ChromaDB semantic search augments the LLM with real movie data
 * - Inference optimizations: SDPA, KV cache, BF16, 4-bit quantization
 *
 * The assistant can:
 * - Recommend movies by genre, mood, actor, director
 * - Find similar movies to ones you enjoyed
 * - Filter by runtime, year, rating
 * - Answer "what should I watch tonight?" style queries
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

    /**
     * Parse the assistant's text response into structured movie cards.
     * The fine-tuned model outputs numbered lists with movie metadata.
     */
    const parseMovies = (text) => {
        if (!text) return []
        const movieBlocks = text.split(/\n(?=\d+\.\s+\*\*)/g).filter(Boolean)
        return movieBlocks.map((block) => {
            const titleMatch = block.match(/\*\*(.+?)\*\*/)
            const yearMatch = block.match(/\((\d{4})\)/)
            const genreMatch = block.match(/\d{4}\)\s*-\s*([^-]+)-/)
            const runtimeMatch = block.match(/(\d+h\s*\d*m|\d+\s*min)/)
            const descLines = block.split('\n').slice(1).join(' ').trim()

            return {
                title: titleMatch?.[1] || 'Unknown',
                year: yearMatch?.[1] || '',
                genre: genreMatch?.[1]?.trim() || '',
                runtime: runtimeMatch?.[0] || '',
                description: descLines.replace(/\*\*.+?\*\*.*?-.*?-.*?\n/, '').trim(),
                raw: block,
            }
        })
    }

    const movies = response ? parseMovies(response.answer || response.response || '') : []

    return (
        <div className="card-cyber max-w-4xl mx-auto">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="text-4xl">🎬</div>
                <div>
                    <h2 className="text-2xl font-bold text-cyber-accent">Movie Discovery</h2>
                    <p className="text-gray-400">
                        Fine-tuned Llama 3 8B · RAG · {' '}
                        <span className="text-cyber-purple text-xs">QLoRA · SDPA · BF16</span>
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
                        {/* Performance metrics */}
                        {(response.query_time_ms || response.retrieval_ms) && (
                            <div className="flex gap-4 text-xs text-gray-400">
                                {response.query_time_ms && <span>⚡ {response.query_time_ms.toFixed(0)}ms total</span>}
                                {response.retrieval_ms && <span>🔍 {response.retrieval_ms.toFixed(0)}ms retrieval</span>}
                                {response.movies_found !== undefined && <span>🎞 {response.movies_found} movies retrieved</span>}
                                {response.backend && <span>🤖 {response.backend}</span>}
                            </div>
                        )}

                        {/* Movie Cards */}
                        {movies.length > 0 ? (
                            <div className="space-y-3">
                                {movies.map((movie, idx) => (
                                    <motion.div
                                        key={idx}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: idx * 0.08 }}
                                        className="p-4 rounded-lg glass-morphism border border-cyber-accent/20
                                                   hover:border-cyber-accent/50 transition-all"
                                    >
                                        <div className="flex items-start justify-between gap-2 mb-1">
                                            <h3 className="font-bold text-white">
                                                {idx + 1}. {movie.title}
                                                {movie.year && (
                                                    <span className="text-gray-400 font-normal ml-2">({movie.year})</span>
                                                )}
                                            </h3>
                                            <div className="flex gap-2 text-xs shrink-0">
                                                {movie.genre && (
                                                    <span className="px-2 py-0.5 rounded-full bg-cyber-purple/20 text-cyber-purple">
                                                        {movie.genre}
                                                    </span>
                                                )}
                                                {movie.runtime && (
                                                    <span className="px-2 py-0.5 rounded-full bg-cyber-accent/10 text-cyber-accent">
                                                        {movie.runtime}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        {movie.description && (
                                            <p className="text-gray-400 text-sm leading-relaxed line-clamp-3">
                                                {movie.description}
                                            </p>
                                        )}
                                    </motion.div>
                                ))}
                            </div>
                        ) : (
                            /* Fallback: raw text response */
                            <div className="p-4 rounded-lg bg-gradient-to-br from-cyber-purple/20 to-cyber-pink/20
                                            border border-cyber-accent/30">
                                <p className="text-sm font-semibold text-cyber-accent mb-2">Recommendations</p>
                                <p className="text-gray-200 leading-relaxed whitespace-pre-wrap text-sm">
                                    {response.answer || response.response || JSON.stringify(response, null, 2)}
                                </p>
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
                        ChromaDB retrieves semantically similar movies, and a fine-tuned{' '}
                        <span className="text-cyber-accent">Llama 3 8B</span> generates personalized
                        recommendations with descriptions. Trained on 2,946 movie Q&A pairs with
                        QLoRA, NEFTune, and rsLoRA.
                    </p>
                </div>
            )}
        </div>
    )
}
