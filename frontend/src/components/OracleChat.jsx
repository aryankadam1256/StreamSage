/**
 * =============================================================================
 * Oracle Chat Component - The Time-Travel Q&A Interface
 * =============================================================================
 * 
 * 🎓 CONCEPT: Stateful Component with API Integration
 * 
 * This component demonstrates:
 * 1. useState hook for local state management
 * 2. Async API calls with fetch/axios
 * 3. Conditional rendering based on loading states
 * 4. Form handling in React
 * 
 * The "Time-Travel" feature allows users to query movie dialogue
 * at specific timestamps, powered by the Oracle RAG service.
 * 
 * =============================================================================
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export default function OracleChat() {
    const [query, setQuery] = useState('')
    const [movieId, setMovieId] = useState('inception')
    const [timestamp, setTimestamp] = useState('')
    const [response, setResponse] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    /**
     * 🎓 ASYNC EVENT HANDLER
     * 
     * Modern React pattern for handling async operations:
     * 1. Prevent default form submission
     * 2. Set loading state
     * 3. Make API call
     * 4. Update state with result or error
     * 5. Clear loading state
     */
    const handleAsk = async (e) => {
        e.preventDefault()

        if (!query.trim()) return

        setLoading(true)
        setError(null)
        setResponse(null)

        try {
            const payload = {
                query: query.trim(),
                movie_id: movieId || undefined,
                timestamp_start: timestamp ? parseFloat(timestamp) : undefined,
                top_k: 5,
            }

            const res = await axios.post(`${API_URL}/oracle/ask`, payload)
            setResponse(res.data)
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to get answer')
        } finally {
            setLoading(false)
        }
    }

    /**
     * Convert seconds to MM:SS format for display
     */
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    return (
        <div className="card-cyber max-w-4xl mx-auto">
            <div className="flex items-center gap-3 mb-6">
                <div className="text-4xl">🔮</div>
                <div>
                    <h2 className="text-2xl font-bold text-cyber-accent">The Oracle</h2>
                    <p className="text-gray-400">Ask about movie scenes with timestamp precision</p>
                </div>
            </div>

            {/* Query Form */}
            <form onSubmit={handleAsk} className="space-y-4 mb-6">
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium mb-2 text-gray-300">
                            Movie ID
                        </label>
                        <input
                            type="text"
                            value={movieId}
                            onChange={(e) => setMovieId(e.target.value)}
                            className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-accent/30 
                         focus:border-cyber-accent focus:outline-none transition-colors"
                            placeholder="e.g., inception"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-2 text-gray-300">
                            Timestamp (seconds) - Optional
                        </label>
                        <input
                            type="number"
                            value={timestamp}
                            onChange={(e) => setTimestamp(e.target.value)}
                            className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-accent/30 
                         focus:border-cyber-accent focus:outline-none transition-colors"
                            placeholder="e.g., 3600 (1 hour)"
                        />
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium mb-2 text-gray-300">
                        Your Question
                    </label>
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        rows={3}
                        className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-accent/30 
                       focus:border-cyber-accent focus:outline-none transition-colors resize-none"
                        placeholder="What did the characters discuss about reality?"
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="btn-cyber w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? (
                        <span className="flex items-center justify-center gap-2">
                            <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                            Consulting the Oracle...
                        </span>
                    ) : (
                        '🔮 Ask Oracle'
                    )}
                </button>
            </form>

            {/* Response Display */}
            <AnimatePresence mode="wait">
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
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
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                    >
                        {/* Answer */}
                        <div className="p-4 rounded-lg bg-gradient-to-br from-cyber-purple/20 to-cyber-pink/20 
                            border border-cyber-accent/30">
                            <p className="text-sm font-semibold text-cyber-accent mb-2">Oracle's Answer</p>
                            <p className="text-gray-200 leading-relaxed">{response.answer}</p>
                            <div className="mt-3 flex gap-4 text-xs text-gray-400">
                                <span>⚡ {response.query_time_ms.toFixed(0)}ms</span>
                                <span>🤖 {response.model_used}</span>
                                <span>📚 {response.sources.length} sources</span>
                            </div>
                        </div>

                        {/* Sources */}
                        {response.sources.length > 0 && (
                            <div>
                                <p className="text-sm font-semibold text-gray-300 mb-3">
                                    📖 Source Chunks ({response.sources.length})
                                </p>
                                <div className="space-y-2 max-h-64 overflow-y-auto">
                                    {response.sources.map((source, idx) => (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: idx * 0.1 }}
                                            className="p-3 rounded-lg glass-morphism text-sm"
                                        >
                                            <div className="flex justify-between items-start mb-2">
                                                <span className="text-cyber-accent font-mono text-xs">
                                                    {formatTime(source.timestamp_start)} - {formatTime(source.timestamp_end)}
                                                </span>
                                                <span className="text-gray-400 text-xs">
                                                    {(source.relevance_score * 100).toFixed(0)}% relevant
                                                </span>
                                            </div>
                                            <p className="text-gray-300">{source.content}</p>
                                        </motion.div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Help Text */}
            {!response && !error && !loading && (
                <div className="mt-6 p-4 rounded-lg glass-morphism">
                    <p className="text-sm text-gray-400">
                        💡 <strong>Tip:</strong> The Oracle uses RAG (Retrieval-Augmented Generation) to answer
                        questions based on actual movie subtitles. Try asking about specific scenes or dialogue!
                    </p>
                </div>
            )}
        </div>
    )
}
