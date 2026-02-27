import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Square, Clock, Copy, Check, ChevronDown, ChevronUp, Loader2, Rewind, FastForward, Eye, EyeOff } from 'lucide-react'
import { askOracleStream, getOracleSuggestions } from '../api'
import ElasticSlider from './ui/ElasticSlider'
import ScrollRevealText from './ui/ScrollRevealText'

function titleToMovieId(title) {
    if (!title) return ''
    return title.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, '_').trim()
}

function formatTime(seconds) {
    if (seconds == null || isNaN(seconds)) return '?'
    const hrs = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const FALLBACK_SUGGESTIONS = [
    "What did the characters discuss about reality?",
    "What happens in the opening scene?",
    "Who is the main character talking to?",
    "What themes does the dialogue explore?",
]

function SourceChunk({ source }) {
    const [expanded, setExpanded] = useState(false)
    const score = (source.relevance_score * 100).toFixed(0)
    return (
        <div className="px-3 py-2.5 rounded-lg bg-brand-card border border-brand-border-subtle text-xs">
            <div className="flex items-center justify-between mb-1.5 gap-2">
                <span className="flex items-center gap-1 text-text-dim font-mono">
                    <Clock size={9} /> {formatTime(source.timestamp_start)} – {formatTime(source.timestamp_end)}
                </span>
                <span className={`badge ${parseInt(score) >= 70 ? 'bg-emerald-500/10 text-emerald-400' : 'badge-neutral'}`}>
                    {score}% match
                </span>
            </div>
            <p className={`text-text-muted leading-relaxed ${expanded ? '' : 'line-clamp-2'}`}>{source.content}</p>
            {source.content.length > 120 && (
                <button onClick={() => setExpanded(v => !v)}
                    className="mt-1.5 text-text-dim hover:text-text-muted transition-colors">
                    {expanded ? 'Show less' : 'Read more'}
                </button>
            )}
        </div>
    )
}

function CopyButton({ text }) {
    const [copied, setCopied] = useState(false)
    const handleCopy = async () => {
        try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) } catch (_) {}
    }
    return (
        <button onClick={handleCopy} className="btn-ghost text-xs" title="Copy">
            {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
    )
}

function OracleAnswer({ entry }) {
    const [showSources, setShowSources] = useState(false)
    const sourceCount = entry.response.sources?.length || 0
    return (
        <div className="bg-brand-surface border border-brand-border-subtle rounded-xl overflow-hidden">
            <div className="px-4 py-3.5">
                <ScrollRevealText text={entry.response.answer}
                    className="text-text-warm text-sm leading-relaxed" />
            </div>
            <div className="flex items-center gap-3 px-4 py-2 border-t border-brand-border-subtle">
                {entry.response.query_time_ms != null && (
                    <span className="text-xs text-text-dim">{Math.round(entry.response.query_time_ms)}ms</span>
                )}
                {sourceCount > 0 && (
                    <button onClick={() => setShowSources(v => !v)}
                        className="flex items-center gap-1 text-xs text-text-muted hover:text-text-warm transition-colors">
                        {showSources ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                        {sourceCount} source{sourceCount !== 1 ? 's' : ''}
                    </button>
                )}
                <div className="ml-auto"><CopyButton text={entry.response.answer} /></div>
            </div>
            <AnimatePresence>
                {showSources && sourceCount > 0 && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }} className="px-3 pb-3 space-y-1.5 overflow-hidden">
                        {entry.response.sources.map((src, i) => <SourceChunk key={i} source={src} />)}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

export default function OracleChat({ movieTitle, movieRuntime }) {
    const [query, setQuery] = useState('')
    const [movieId, setMovieId] = useState(titleToMovieId(movieTitle) || '')
    const [timestamp, setTimestamp] = useState(0)
    const [alreadyWatched, setAlreadyWatched] = useState(false)
    const [chatHistory, setChatHistory] = useState([])
    const [streaming, setStreaming] = useState(false)
    const [streamTokens, setStreamTokens] = useState('')
    const [streamSources, setStreamSources] = useState([])
    const [error, setError] = useState(null)
    const [suggestions, setSuggestions] = useState(FALLBACK_SUGGESTIONS)

    const abortRef = useRef(null)
    const bottomRef = useRef(null)
    const streamTokensRef = useRef('')
    const streamSourcesRef = useRef([])

    useEffect(() => {
        if (!movieId) return
        getOracleSuggestions(movieId).then(s => { if (s?.length) setSuggestions(s) })
    }, [movieId])

    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatHistory, streamTokens])
    useEffect(() => { streamTokensRef.current = streamTokens }, [streamTokens])
    useEffect(() => { streamSourcesRef.current = streamSources }, [streamSources])

    const buildHistory = useCallback(() =>
        chatHistory.flatMap(entry => [
            { role: 'user', content: entry.query },
            { role: 'oracle', content: entry.response.answer },
        ]), [chatHistory])

    const handleAsk = (e) => {
        e?.preventDefault()
        const q = query.trim()
        if (!q || !movieId.trim() || streaming) return
        setStreaming(true); setStreamTokens(''); setStreamSources([]); setError(null)
        const ts = !alreadyWatched && timestamp > 0 ? String(timestamp) : undefined
        const controller = askOracleStream(q, movieId, ts, buildHistory(), alreadyWatched, {
            onSources: (sources) => setStreamSources(sources),
            onToken: (content) => setStreamTokens(prev => prev + content),
            onDone: (meta) => {
                setChatHistory(prev => [...prev, {
                    query: q,
                    response: { answer: streamTokensRef.current, sources: streamSourcesRef.current,
                        model_used: meta.model_used, query_time_ms: meta.query_time_ms },
                }])
                setStreamTokens(''); setStreamSources([]); setStreaming(false); setQuery('')
            },
            onError: (msg) => { setError(msg || 'Stream failed'); setStreaming(false); setStreamTokens('') },
        })
        abortRef.current = controller
    }

    const handleStop = () => {
        abortRef.current?.abort()
        if (streamTokensRef.current) {
            setChatHistory(prev => [...prev, {
                query: query.trim(),
                response: { answer: streamTokensRef.current + ' [stopped]', sources: streamSourcesRef.current,
                    model_used: 'stopped', query_time_ms: null },
            }])
        }
        setStreaming(false); setStreamTokens(''); setStreamSources([]); setQuery('')
    }

    const showEmpty = chatHistory.length === 0 && !streaming && !error

    return (
        <div className="max-w-3xl mx-auto space-y-4">
            {/* Controls */}
            <div className="bg-brand-surface border border-brand-border-subtle rounded-xl p-4 space-y-4">
                <div>
                    <label className="block text-xs text-text-muted mb-1.5">Movie ID</label>
                    <input type="text" value={movieId} onChange={(e) => setMovieId(e.target.value)}
                        className="input-field text-sm py-2" placeholder="e.g. inception" />
                </div>
                {/* Elastic Slider for timestamp */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <label className="text-xs text-text-muted">
                            {alreadyWatched
                                ? 'Timestamp — disabled (already watched)'
                                : `Timestamp — ${timestamp > 0 ? formatTime(timestamp) : 'any scene'}`}
                        </label>
                        <button
                            onClick={() => setAlreadyWatched(v => !v)}
                            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border transition-all duration-200 ${
                                alreadyWatched
                                    ? 'bg-brand-gold/15 border-brand-gold/40 text-brand-gold'
                                    : 'bg-brand-card border-brand-border-subtle text-text-dim hover:text-text-muted hover:border-brand-border'
                            }`}
                            title={alreadyWatched ? 'Switch to watching mode' : 'Mark as already watched — no spoiler protection'}
                        >
                            {alreadyWatched ? <Eye size={11} /> : <EyeOff size={11} />}
                            Already watched
                        </button>
                    </div>
                    <div className={alreadyWatched ? 'opacity-30 pointer-events-none select-none' : ''}>
                        <ElasticSlider
                            defaultValue={0} startingValue={0} maxValue={(movieRuntime || 120) * 60}
                            isStepped={true} stepSize={30}
                            leftIcon={<Rewind size={16} />}
                            rightIcon={<FastForward size={16} />}
                            onChange={(v) => setTimestamp(v)}
                        />
                    </div>
                    {alreadyWatched && (
                        <p className="mt-1.5 text-xs text-text-dim">
                            Full movie knowledge — The Oracle can discuss everything freely.
                        </p>
                    )}
                </div>
            </div>

            {/* Chat area */}
            <div className="bg-brand-surface border border-brand-border-subtle rounded-xl overflow-hidden">
                <div className="p-4 space-y-4 max-h-[480px] overflow-y-auto">
                    {showEmpty && (
                        <div className="py-4">
                            <p className="section-label mb-3 text-center">Try asking</p>
                            <div className="flex flex-wrap gap-2 justify-center">
                                {suggestions.map((sq, i) => (
                                    <button key={i} onClick={() => setQuery(sq)}
                                        className="px-3 py-1.5 text-xs text-text-dim rounded-lg
                                                   bg-brand-card border border-brand-border-subtle
                                                   hover:border-brand-gold/20 hover:text-text-muted
                                                   transition-all duration-150 text-left">
                                        {sq}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    <AnimatePresence initial={false}>
                        {chatHistory.map((entry, idx) => (
                            <motion.div key={idx} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                                className="space-y-3">
                                <div className="flex justify-end">
                                    <div className="max-w-[78%] px-3.5 py-2.5 rounded-xl bg-brand-gold/10
                                                    border border-brand-gold/20 text-text-warm text-sm">
                                        {entry.query}
                                    </div>
                                </div>
                                <OracleAnswer entry={entry} />
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {streaming && (
                        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                            <div className="flex justify-end">
                                <div className="max-w-[78%] px-3.5 py-2.5 rounded-xl bg-brand-gold/10
                                                border border-brand-gold/20 text-text-warm text-sm">
                                    {query || '...'}
                                </div>
                            </div>
                            <div className="bg-brand-surface border border-brand-border-subtle rounded-xl px-4 py-3.5">
                                {streamTokens ? (
                                    <p className="text-text-warm text-sm leading-relaxed whitespace-pre-wrap">
                                        {streamTokens}
                                        <span className="inline-block w-1.5 h-4 ml-0.5 bg-brand-gold animate-pulse
                                                          align-middle rounded-sm" />
                                    </p>
                                ) : (
                                    <div className="flex items-center gap-2 text-text-muted text-sm">
                                        <Loader2 size={13} className="animate-spin" /> Retrieving context...
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {error && (
                        <div className="px-3 py-2.5 rounded-lg bg-brand-crimson/10 border border-brand-crimson/20
                                        text-brand-crimson text-sm">{error}</div>
                    )}
                    <div ref={bottomRef} />
                </div>

                {/* Input row */}
                <div className="border-t border-brand-border-subtle p-3 flex gap-2">
                    <input type="text" value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleAsk(e)}
                        className="flex-1 bg-brand-card border border-brand-border rounded-lg px-3 py-2.5
                                   text-sm text-text-warm placeholder:text-text-dim
                                   focus:outline-none focus:border-brand-gold/30 transition-colors"
                        placeholder="Ask about scenes, dialogue, characters..."
                        disabled={streaming}
                    />
                    {streaming ? (
                        <button onClick={handleStop} className="btn-secondary text-sm px-3">
                            <Square size={13} className="fill-current" /> Stop
                        </button>
                    ) : (
                        <button onClick={handleAsk} disabled={!query.trim() || !movieId.trim()}
                            className="btn-primary text-sm px-4 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none">
                            <Send size={13} /> Ask
                        </button>
                    )}
                    {chatHistory.length > 0 && !streaming && (
                        <button onClick={() => { setChatHistory([]); setError(null) }}
                            className="btn-ghost text-xs px-2">Clear</button>
                    )}
                </div>
            </div>

            {showEmpty && (
                <p className="text-xs text-text-dim text-center">
                    Answers stream in real-time from subtitle embeddings with timestamp citations.
                </p>
            )}
        </div>
    )
}
