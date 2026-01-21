/**
 * =============================================================================
 * VibeBar Component - Sentiment Visualization
 * =============================================================================
 * 
 * This component demonstrates sentiment analysis using the BERT-powered
 * sentiment service. Users can input movie reviews and see real-time
 * sentiment analysis with confidence scores.
 * 
 * =============================================================================
 */

import { useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export default function VibeBar() {
    const [text, setText] = useState('')
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleAnalyze = async (e) => {
        e.preventDefault()

        if (!text.trim()) return

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await axios.post(`${API_URL}/sentiment/analyze`, {
                text: text.trim(),
                include_explanation: true,
            })
            setResult(res.data)
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Analysis failed')
        } finally {
            setLoading(false)
        }
    }

    /**
     * Sample reviews for quick testing
     */
    const samples = [
        "This movie was absolutely amazing! The cinematography was breathtaking.",
        "Terrible acting and a completely predictable plot. Total waste of time.",
        "It was okay. Nothing special but not bad either.",
    ]

    return (
        <div className="card-cyber max-w-4xl mx-auto">
            <div className="flex items-center gap-3 mb-6">
                <div className="text-4xl">💬</div>
                <div>
                    <h2 className="text-2xl font-bold text-cyber-purple">Sentiment Vibe Bar</h2>
                    <p className="text-gray-400">Analyze movie review sentiment with BERT</p>
                </div>
            </div>

            {/* Input Form */}
            <form onSubmit={handleAnalyze} className="space-y-4 mb-6">
                <div>
                    <label className="block text-sm font-medium mb-2 text-gray-300">
                        Movie Review Text
                    </label>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        rows={4}
                        className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-purple/30 
                       focus:border-cyber-purple focus:outline-none transition-colors resize-none"
                        placeholder="Enter a movie review to analyze its sentiment..."
                    />
                </div>

                {/* Sample Buttons */}
                <div className="flex gap-2 flex-wrap">
                    <span className="text-xs text-gray-400 self-center">Quick samples:</span>
                    {samples.map((sample, idx) => (
                        <button
                            key={idx}
                            type="button"
                            onClick={() => setText(sample)}
                            className="text-xs px-3 py-1 rounded glass-morphism hover:bg-white/10 transition-colors"
                        >
                            Sample {idx + 1}
                        </button>
                    ))}
                </div>

                <button
                    type="submit"
                    disabled={loading || !text.trim()}
                    className="btn-cyber w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? (
                        <span className="flex items-center justify-center gap-2">
                            <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                            Analyzing Sentiment...
                        </span>
                    ) : (
                        '💬 Analyze Sentiment'
                    )}
                </button>
            </form>

            {/* Error Display */}
            {error && (
                <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 mb-6">
                    <p className="font-semibold">Error</p>
                    <p className="text-sm">{error}</p>
                </div>
            )}

            {/* Result Display */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-6"
                >
                    {/* Sentiment Label */}
                    <div className="text-center">
                        <div className={`inline-block px-6 py-3 rounded-full text-2xl font-bold ${result.sentiment.label === 'positive'
                                ? 'bg-green-500/20 text-green-400 border-2 border-green-500'
                                : 'bg-red-500/20 text-red-400 border-2 border-red-500'
                            }`}>
                            {result.sentiment.label === 'positive' ? '😊 Positive' : '😞 Negative'}
                        </div>
                        <p className="mt-2 text-sm text-gray-400">{result.explanation}</p>
                    </div>

                    {/* Confidence Bar */}
                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400">Confidence</span>
                            <span className="font-mono text-cyber-accent">
                                {(result.sentiment.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="h-3 bg-cyber-bg rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${result.sentiment.confidence * 100}%` }}
                                transition={{ duration: 0.8, ease: "easeOut" }}
                                className={`h-full ${result.sentiment.label === 'positive'
                                        ? 'bg-gradient-to-r from-green-500 to-emerald-400'
                                        : 'bg-gradient-to-r from-red-500 to-pink-400'
                                    }`}
                            />
                        </div>
                    </div>

                    {/* Score Breakdown */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 rounded-lg glass-morphism">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-gray-400">😊 Positive</span>
                                <span className="font-mono text-green-400">
                                    {(result.sentiment.scores.positive * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="h-2 bg-cyber-bg rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-green-500"
                                    style={{ width: `${result.sentiment.scores.positive * 100}%` }}
                                />
                            </div>
                        </div>

                        <div className="p-4 rounded-lg glass-morphism">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-gray-400">😞 Negative</span>
                                <span className="font-mono text-red-400">
                                    {(result.sentiment.scores.negative * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="h-2 bg-cyber-bg rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-red-500"
                                    style={{ width: `${result.sentiment.scores.negative * 100}%` }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Review Preview */}
                    <div className="p-4 rounded-lg bg-cyber-card border border-cyber-purple/20">
                        <p className="text-xs text-gray-400 mb-2">Analyzed Text</p>
                        <p className="text-sm text-gray-300 italic">&ldquo;{result.text}&rdquo;</p>
                    </div>
                </motion.div>
            )}

            {/* Info */}
            {!result && !error && !loading && (
                <div className="mt-6 p-4 rounded-lg glass-morphism">
                    <p className="text-sm text-gray-400">
                        🧠 <strong>How it works:</strong> This uses a BERT transformer model fine-tuned on
                        movie reviews to detect sentiment. BERT understands context bidirectionally, catching
                        nuances like sarcasm and negation.
                    </p>
                </div>
            )}
        </div>
    )
}
