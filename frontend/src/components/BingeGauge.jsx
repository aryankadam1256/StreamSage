/**
 * =============================================================================
 * BingeGauge Component - Watch Pattern Prediction
 * =============================================================================
 * 
 * This component visualizes binge-watching predictions using the LSTM-powered
 * binge predictor service. It displays a speedometer-style gauge showing the
 * probability that a user will continue watching.
 * 
 * =============================================================================
 */

import { useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export default function BingeGauge() {
    const [userId, setUserId] = useState('user_123')
    const [watchHistory, setWatchHistory] = useState([
        { movie_id: 'm1', genre_ids: [28], rating: 8.5, watch_duration_pct: 1.0, timestamp: Date.now() / 1000 - 7200 },
        { movie_id: 'm2', genre_ids: [35], rating: 7.5, watch_duration_pct: 1.0, timestamp: Date.now() / 1000 - 3600 },
        { movie_id: 'm3', genre_ids: [28], rating: 9.0, watch_duration_pct: 1.0, timestamp: Date.now() / 1000 - 1800 },
    ])
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handlePredict = async () => {
        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await axios.post(`${API_URL}/binge/predict`, {
                user_id: userId,
                watch_history: watchHistory,
                current_hour: new Date().getHours(),
            })
            setResult(res.data)
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Prediction failed')
        } finally {
            setLoading(false)
        }
    }

    /**
     * Add a movie to watch history
     */
    const addMovie = () => {
        setWatchHistory([
            ...watchHistory,
            {
                movie_id: `m${watchHistory.length + 1}`,
                genre_ids: [Math.floor(Math.random() * 50)],
                rating: Math.random() * 5 + 5, // 5-10
                watch_duration_pct: Math.random() * 0.3 + 0.7, // 0.7-1.0
                timestamp: Date.now() / 1000,
            },
        ])
    }

    /**
     * Get gauge color based on probability
     */
    const getGaugeColor = (prob) => {
        if (prob >= 0.7) return '#10b981' // green
        if (prob >= 0.4) return '#f59e0b' // orange
        return '#ef4444' // red
    }

    /**
     * Calculate gauge rotation (0-180 degrees)
     */
    const getRotation = (prob) => {
        return prob * 180
    }

    return (
        <div className="card-cyber max-w-4xl mx-auto">
            <div className="flex items-center gap-3 mb-6">
                <div className="text-4xl">📊</div>
                <div>
                    <h2 className="text-2xl font-bold text-cyber-pink">Binge Gauge</h2>
                    <p className="text-gray-400">Predict watch continuation with LSTM</p>
                </div>
            </div>

            {/* Controls */}
            <div className="space-y-4 mb-6">
                <div>
                    <label className="block text-sm font-medium mb-2 text-gray-300">
                        User ID
                    </label>
                    <input
                        type="text"
                        value={userId}
                        onChange={(e) => setUserId(e.target.value)}
                        className="w-full px-4 py-2 rounded-lg bg-cyber-bg border border-cyber-pink/30 
                       focus:border-cyber-pink focus:outline-none transition-colors"
                    />
                </div>

                {/* Watch History */}
                <div>
                    <div className="flex justify-between items-center mb-2">
                        <label className="block text-sm font-medium text-gray-300">
                            Watch History ({watchHistory.length} movies)
                        </label>
                        <button
                            type="button"
                            onClick={addMovie}
                            className="text-xs px-3 py-1 rounded glass-morphism hover:bg-white/10 transition-colors"
                        >
                            + Add Movie
                        </button>
                    </div>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                        {watchHistory.map((movie, idx) => (
                            <div key={idx} className="p-3 rounded-lg glass-morphism text-sm flex justify-between">
                                <div>
                                    <span className="font-mono text-cyber-accent">{movie.movie_id}</span>
                                    <span className="text-gray-400 ml-3">Rating: {movie.rating.toFixed(1)}</span>
                                </div>
                                <span className="text-gray-400">{(movie.watch_duration_pct * 100).toFixed(0)}% watched</span>
                            </div>
                        ))}
                    </div>
                </div>

                <button
                    onClick={handlePredict}
                    disabled={loading || watchHistory.length === 0}
                    className="btn-cyber w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? (
                        <span className="flex items-center justify-center gap-2">
                            <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                            Analyzing Pattern...
                        </span>
                    ) : (
                        '📊 Predict Binge Probability'
                    )}
                </button>
            </div>

            {/* Error */}
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
                    {/* Speedometer Gauge */}
                    <div className="relative flex justify-center items-center py-8">
                        <svg width="300" height="180" viewBox="0 0 300 180">
                            {/* Background arc */}
                            <path
                                d="M 30 150 A 120 120 0 0 1 270 150"
                                fill="none"
                                stroke="#1a1f3a"
                                strokeWidth="30"
                                strokeLinecap="round"
                            />

                            {/* Colored arc (animated) */}
                            <motion.path
                                d="M 30 150 A 120 120 0 0 1 270 150"
                                fill="none"
                                stroke={getGaugeColor(result.continue_probability)}
                                strokeWidth="30"
                                strokeLinecap="round"
                                strokeDasharray="377" // Circumference
                                initial={{ strokeDashoffset: 377 }}
                                animate={{
                                    strokeDashoffset: 377 - (377 * result.continue_probability)
                                }}
                                transition={{ duration: 1.5, ease: "easeOut" }}
                            />

                            {/* Needle */}
                            <motion.g
                                initial={{ rotate: 0 }}
                                animate={{ rotate: getRotation(result.continue_probability) }}
                                transition={{ duration: 1.5, ease: "easeOut" }}
                                style={{ transformOrigin: '150px 150px' }}
                            >
                                <line
                                    x1="150"
                                    y1="150"
                                    x2="150"
                                    y2="50"
                                    stroke="#00f0ff"
                                    strokeWidth="3"
                                    strokeLinecap="round"
                                />
                                <circle cx="150" cy="150" r="8" fill="#00f0ff" />
                            </motion.g>

                            {/* Center text */}
                            <text
                                x="150"
                                y="140"
                                textAnchor="middle"
                                fill="#fff"
                                fontSize="32"
                                fontWeight="bold"
                            >
                                {(result.continue_probability * 100).toFixed(0)}%
                            </text>
                        </svg>

                        <div className="absolute bottom-0 text-center">
                            <p className="text-sm text-gray-400">Continue Probability</p>
                        </div>
                    </div>

                    {/* Risk Level */}
                    <div className={`p-4 rounded-lg text-center ${result.risk_level === 'low'
                            ? 'bg-green-500/20 border-2 border-green-500'
                            : result.risk_level === 'medium'
                                ? 'bg-yellow-500/20 border-2 border-yellow-500'
                                : 'bg-red-500/20 border-2 border-red-500'
                        }`}>
                        <p className="text-sm text-gray-400 mb-1">Drop-off Risk</p>
                        <p className="text-2xl font-bold uppercase">{result.risk_level}</p>
                    </div>

                    {/* Recommendation */}
                    <div className="p-4 rounded-lg bg-gradient-to-br from-cyber-purple/20 to-cyber-pink/20 
                          border border-cyber-accent/30">
                        <p className="text-sm font-semibold text-cyber-accent mb-2">📋 Recommendation</p>
                        <p className="text-gray-200">{result.recommendation}</p>
                    </div>

                    {/* Model Info */}
                    <div className="text-center text-xs text-gray-400">
                        Model: {result.model_version} · User: {result.user_id}
                    </div>
                </motion.div>
            )}

            {/* Info */}
            {!result && !error && !loading && (
                <div className="mt-6 p-4 rounded-lg glass-morphism">
                    <p className="text-sm text-gray-400">
                        🎯 <strong>How it works:</strong> This uses an LSTM (Long Short-Term Memory) model
                        to analyze sequential watch patterns. The model learns from past viewing behavior to
                        predict future engagement. Features include rating trends, watch velocity, genre
                        preferences, and temporal patterns.
                    </p>
                </div>
            )}
        </div>
    )
}
