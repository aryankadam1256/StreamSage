import { useState } from 'react'
import { motion } from 'framer-motion'
import { Plus, Trash2, Loader2, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

function getGaugeColor(prob) {
    if (prob >= 0.7) return '#10b981'
    if (prob >= 0.4) return '#d4a017'
    return '#c0103a'
}

function RiskIcon({ level }) {
    if (level === 'low') return <TrendingUp size={14} className="text-emerald-400" />
    if (level === 'high') return <TrendingDown size={14} className="text-brand-crimson" />
    return <Minus size={14} className="text-brand-gold" />
}

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

    const addMovie = () => {
        setWatchHistory(prev => [...prev, {
            movie_id: `m${prev.length + 1}`,
            genre_ids: [Math.floor(Math.random() * 50)],
            rating: +(Math.random() * 5 + 5).toFixed(1),
            watch_duration_pct: +(Math.random() * 0.3 + 0.7).toFixed(2),
            timestamp: Date.now() / 1000,
        }])
    }

    const removeMovie = (idx) => {
        setWatchHistory(prev => prev.filter((_, i) => i !== idx))
    }

    const prob = result?.continue_probability ?? 0
    const gaugeColor = getGaugeColor(prob)
    const needleRotation = prob * 180 - 90

    const riskStyles = {
        low: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
        medium: 'bg-brand-gold/10 border-brand-gold/20 text-brand-gold',
        high: 'bg-brand-crimson/10 border-brand-crimson/20 text-brand-crimson',
    }

    return (
        <div className="max-w-3xl mx-auto space-y-4">
            {/* Config card */}
            <div className="bg-brand-surface border border-brand-border-subtle rounded-xl p-5">
                <p className="section-label mb-4">Configuration</p>

                {/* User ID */}
                <div className="mb-4">
                    <label className="block text-xs text-text-muted mb-1.5">User ID</label>
                    <input
                        type="text"
                        value={userId}
                        onChange={(e) => setUserId(e.target.value)}
                        className="input-field text-sm py-2"
                    />
                </div>

                {/* Watch history */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <label className="text-xs text-text-muted">
                            Watch history
                            <span className="ml-1.5 badge-neutral">{watchHistory.length}</span>
                        </label>
                        <button onClick={addMovie} className="btn-ghost text-xs">
                            <Plus size={12} />
                            <span>Add movie</span>
                        </button>
                    </div>
                    <div className="space-y-1.5 max-h-44 overflow-y-auto rounded-lg">
                        {watchHistory.map((movie, idx) => (
                            <div
                                key={idx}
                                className="flex items-center justify-between px-3 py-2 rounded-lg
                                           bg-brand-card border border-brand-border-subtle text-xs"
                            >
                                <div className="flex items-center gap-3 min-w-0">
                                    <span className="font-mono text-text-warm shrink-0">
                                        {movie.movie_id}
                                    </span>
                                    <span className="text-text-dim">
                                        ★ {movie.rating.toFixed(1)}
                                    </span>
                                    <span className="text-text-dim">
                                        {(movie.watch_duration_pct * 100).toFixed(0)}% watched
                                    </span>
                                </div>
                                {watchHistory.length > 1 && (
                                    <button
                                        onClick={() => removeMovie(idx)}
                                        className="text-text-dim hover:text-brand-crimson transition-colors ml-2 shrink-0"
                                    >
                                        <Trash2 size={11} />
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                <button
                    onClick={handlePredict}
                    disabled={loading || watchHistory.length === 0}
                    className="btn-primary w-full justify-center mt-4 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
                >
                    {loading
                        ? <><Loader2 size={15} className="animate-spin" /> Analyzing…</>
                        : 'Predict Binge Probability'
                    }
                </button>
            </div>

            {/* Error */}
            {error && (
                <div className="px-4 py-3 rounded-xl bg-brand-crimson/10 border border-brand-crimson/20 text-brand-crimson text-sm">
                    {error}
                </div>
            )}

            {/* Results */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-brand-surface border border-brand-border-subtle rounded-xl overflow-hidden"
                >
                    {/* Gauge */}
                    <div className="flex flex-col items-center pt-8 pb-4 px-4">
                        <svg width="280" height="150" viewBox="0 0 280 150">
                            {/* Track arc */}
                            <path
                                d="M 20 130 A 120 120 0 0 1 260 130"
                                fill="none"
                                stroke="#161622"
                                strokeWidth="24"
                                strokeLinecap="round"
                            />
                            {/* Filled arc */}
                            <motion.path
                                d="M 20 130 A 120 120 0 0 1 260 130"
                                fill="none"
                                stroke={gaugeColor}
                                strokeWidth="24"
                                strokeLinecap="round"
                                strokeDasharray="377"
                                initial={{ strokeDashoffset: 377 }}
                                animate={{ strokeDashoffset: 377 - 377 * prob }}
                                transition={{ duration: 1.2, ease: 'easeOut' }}
                            />
                            {/* Needle pivot */}
                            <motion.line
                                x1="140" y1="130" x2="140" y2="40"
                                stroke="#e8e0d0"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                initial={{ rotate: -90 }}
                                animate={{ rotate: needleRotation }}
                                transition={{ duration: 1.2, ease: 'easeOut' }}
                                style={{ transformOrigin: '140px 130px' }}
                            />
                            <circle cx="140" cy="130" r="6" fill="#e8e0d0" />
                            {/* Probability text */}
                            <text
                                x="140" y="115"
                                textAnchor="middle"
                                fill="#e8e0d0"
                                fontSize="28"
                                fontWeight="700"
                                fontFamily="Inter, system-ui, sans-serif"
                            >
                                {(prob * 100).toFixed(0)}%
                            </text>
                        </svg>
                        <p className="text-xs text-text-muted -mt-1">Continue probability</p>
                    </div>

                    {/* Risk + Recommendation */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-brand-border-subtle border-t border-brand-border-subtle">
                        {/* Risk level */}
                        <div className="p-5">
                            <p className="section-label mb-3">Drop-off Risk</p>
                            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg
                                            border text-sm font-semibold ${riskStyles[result.risk_level] || riskStyles.medium}`}>
                                <RiskIcon level={result.risk_level} />
                                <span className="capitalize">{result.risk_level}</span>
                            </div>
                        </div>

                        {/* Recommendation */}
                        <div className="p-5">
                            <p className="section-label mb-3">Recommendation</p>
                            <p className="text-sm text-text-warm leading-relaxed">
                                {result.recommendation}
                            </p>
                        </div>
                    </div>

                    {/* Footer meta */}
                    <div className="flex items-center justify-between px-5 py-2.5 border-t border-brand-border-subtle">
                        <span className="text-xs text-text-dim">
                            Model: {result.model_version}
                        </span>
                        <span className="text-xs text-text-dim">
                            User: {result.user_id}
                        </span>
                    </div>
                </motion.div>
            )}

            {/* Info state */}
            {!result && !error && !loading && (
                <div className="px-4 py-3 rounded-xl bg-brand-surface border border-brand-border-subtle">
                    <p className="text-xs text-text-muted leading-relaxed">
                        Uses an LSTM (Long Short-Term Memory) model to analyze sequential watch patterns.
                        Features include rating trends, watch velocity, genre preferences, and temporal patterns.
                    </p>
                </div>
            )}
        </div>
    )
}
