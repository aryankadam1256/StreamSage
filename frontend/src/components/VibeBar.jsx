import { useState } from 'react'
import { motion } from 'framer-motion'
import { Loader2, ThumbsUp, ThumbsDown, Minus } from 'lucide-react'
import { analyzeSentiment } from '../api'

export default function VibeBar({ movieTitle }) {
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
            const data = await analyzeSentiment(text)
            setResult(data)
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Analysis failed')
        } finally {
            setLoading(false)
        }
    }

    const samples = movieTitle
        ? [
            `${movieTitle} was absolutely stunning — the performances were unforgettable.`,
            `${movieTitle} was a total disappointment. Predictable and boring.`,
            `${movieTitle} was decent. Has its moments but nothing groundbreaking.`,
        ]
        : [
            "Incredible cinematography and a script that keeps you guessing till the last frame.",
            "Terrible acting and a completely predictable plot. Total waste of time.",
            "It was okay. Some good moments but overall just average.",
        ]

    const isPositive = result?.sentiment?.label === 'positive'
    const confidence = result ? (result.sentiment.confidence * 100).toFixed(1) : 0

    return (
        <div className="max-w-3xl mx-auto space-y-4">
            {/* Input card */}
            <div className="bg-brand-surface border border-brand-border-subtle rounded-xl p-5">
                <p className="section-label mb-4">Review Text</p>

                <form onSubmit={handleAnalyze} className="space-y-3">
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        rows={4}
                        className="w-full bg-brand-card border border-brand-border rounded-lg px-4 py-3
                                   text-text-warm text-sm placeholder:text-text-dim resize-none
                                   focus:outline-none focus:border-brand-gold/30 transition-colors"
                        placeholder={movieTitle
                            ? `Write a review for ${movieTitle}…`
                            : "Paste a movie review to analyze its sentiment…"}
                    />

                    {/* Sample buttons */}
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs text-text-dim">Try a sample:</span>
                        {samples.map((sample, idx) => (
                            <button
                                key={idx}
                                type="button"
                                onClick={() => setText(sample)}
                                className="text-xs px-2.5 py-1 rounded-lg bg-brand-card
                                           border border-brand-border-subtle text-text-muted
                                           hover:border-brand-gold/20 hover:text-text-warm
                                           transition-all duration-150"
                            >
                                {idx === 0 ? 'Positive' : idx === 1 ? 'Negative' : 'Neutral'}
                            </button>
                        ))}
                    </div>

                    <button
                        type="submit"
                        disabled={loading || !text.trim()}
                        className="btn-primary w-full justify-center disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
                    >
                        {loading
                            ? <><Loader2 size={15} className="animate-spin" /> Analyzing…</>
                            : 'Analyze Sentiment'}
                    </button>
                </form>
            </div>

            {/* Error */}
            {error && (
                <div className="px-4 py-3 rounded-xl bg-brand-crimson/10 border border-brand-crimson/20 text-brand-crimson text-sm">
                    <p className="font-medium">Analysis failed</p>
                    <p className="text-xs mt-0.5 opacity-70">{error}</p>
                </div>
            )}

            {/* Results card */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-brand-surface border border-brand-border-subtle rounded-xl overflow-hidden"
                >
                    {/* Verdict row */}
                    <div className="flex items-center gap-4 px-5 py-4 border-b border-brand-border-subtle">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
                            isPositive
                                ? 'bg-emerald-500/10 text-emerald-400'
                                : 'bg-brand-crimson/10 text-brand-crimson'
                        }`}>
                            {isPositive
                                ? <ThumbsUp size={18} />
                                : <ThumbsDown size={18} />
                            }
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-1.5">
                                <span className={`text-base font-semibold ${
                                    isPositive ? 'text-emerald-400' : 'text-brand-crimson'
                                }`}>
                                    {isPositive ? 'Positive' : 'Negative'}
                                </span>
                                <span className="text-sm font-mono text-text-warm">{confidence}%</span>
                            </div>
                            {/* Confidence bar */}
                            <div className="h-1.5 bg-brand-card rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${confidence}%` }}
                                    transition={{ duration: 0.7, ease: 'easeOut' }}
                                    className={`h-full rounded-full ${
                                        isPositive
                                            ? 'bg-emerald-500'
                                            : 'bg-brand-crimson'
                                    }`}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Score breakdown */}
                    <div className="grid grid-cols-2 divide-x divide-brand-border-subtle">
                        {[
                            {
                                label: 'Positive',
                                score: result.sentiment.scores.positive,
                                Icon: ThumbsUp,
                                color: 'text-emerald-400',
                                bar: 'bg-emerald-500',
                            },
                            {
                                label: 'Negative',
                                score: result.sentiment.scores.negative,
                                Icon: ThumbsDown,
                                color: 'text-brand-crimson',
                                bar: 'bg-brand-crimson',
                            },
                        ].map(({ label, score, Icon, color, bar }) => (
                            <div key={label} className="px-5 py-4">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="flex items-center gap-1.5 text-xs text-text-dim">
                                        <Icon size={11} />
                                        {label}
                                    </span>
                                    <span className={`text-sm font-mono font-semibold ${color}`}>
                                        {(score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="h-1 bg-brand-card rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full ${bar}`}
                                        style={{ width: `${score * 100}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Explanation */}
                    {result.explanation && (
                        <div className="px-5 py-3 border-t border-brand-border-subtle">
                            <p className="text-xs text-text-muted leading-relaxed">{result.explanation}</p>
                        </div>
                    )}

                    {/* Analyzed text */}
                    <div className="px-5 py-3 border-t border-brand-border-subtle bg-brand-card/40">
                        <p className="section-label mb-1.5">Input Text</p>
                        <p className="text-xs text-text-muted italic leading-relaxed line-clamp-3">
                            "{result.text}"
                        </p>
                    </div>
                </motion.div>
            )}

            {/* Info state */}
            {!result && !error && !loading && (
                <div className="px-4 py-3 rounded-xl bg-brand-surface border border-brand-border-subtle">
                    <p className="text-xs text-text-muted leading-relaxed">
                        Uses a BERT transformer fine-tuned on movie reviews to classify sentiment.
                        BERT reads context bidirectionally, catching nuances like sarcasm and negation.
                    </p>
                </div>
            )}
        </div>
    )
}
