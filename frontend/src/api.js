import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const OMDB_API_URL = 'https://www.omdbapi.com/'
const OMDB_API_KEY = import.meta.env.VITE_OMDB_API_KEY || 'thewdb'
const movieImageCache = new Map()
const movieImageInFlight = new Map()
let tmdbCircuitOpenUntil = 0
const TMDB_CIRCUIT_COOLDOWN_MS = 30 * 1000
const DEFAULT_IMAGE_PAYLOAD = {
    poster_path: null,
    backdrop_path: null,
    posters: [],
    backdrops: [],
    tmdb_available: false,
}

function dedupeUrls(list = []) {
    const seen = new Set()
    const result = []
    for (const item of list) {
        if (!item || seen.has(item)) continue
        seen.add(item)
        result.push(item)
    }
    return result
}

function normalizeImagePayload(data) {
    const base = data || DEFAULT_IMAGE_PAYLOAD
    const posters = dedupeUrls([...(Array.isArray(base.posters) ? base.posters : []), base.poster_path])
    const backdrops = dedupeUrls([...(Array.isArray(base.backdrops) ? base.backdrops : []), base.backdrop_path])
    return {
        ...base,
        poster_path: posters[0] || null,
        backdrop_path: backdrops[0] || null,
        posters,
        backdrops,
    }
}

async function fetchOmdbPoster(title, year) {
    if (!title || !OMDB_API_KEY) return null
    try {
        const res = await axios.get(OMDB_API_URL, {
            params: {
                t: title,
                y: year || undefined,
                apikey: OMDB_API_KEY,
            },
            timeout: 5000,
        })
        const poster = res?.data?.Poster
        return poster && poster !== 'N/A' ? poster : null
    } catch (_) {
        return null
    }
}

export async function discoverMovies(query, filters = {}) {
    const payload = {
        query: query.trim(),
        top_k: filters.top_k || 5,
    }
    if (filters.genre) payload.genre = filters.genre
    if (filters.min_rating) payload.min_rating = parseFloat(filters.min_rating)
    if (filters.min_year) payload.min_year = parseInt(filters.min_year)
    if (filters.max_year) payload.max_year = parseInt(filters.max_year)

    const res = await axios.post(`${API_URL}/discover`, payload)
    return res.data
}

export async function askOracle(query, movieId, timestamp, conversationHistory = [], alreadyWatched = false) {
    const payload = {
        query: query.trim(),
        movie_id: movieId || undefined,
        timestamp: timestamp ? parseFloat(timestamp) : undefined,
        top_k: 5,
        conversation_history: conversationHistory,
        already_watched: alreadyWatched || false,
    }
    const res = await axios.post(`${API_URL}/oracle/ask`, payload)
    return res.data
}

/**
 * Stream the Oracle's answer via Server-Sent Events (SSE).
 *
 * Uses fetch() + ReadableStream rather than EventSource because EventSource
 * only supports GET requests and we need to POST the JSON body.
 *
 * @param {string} query - The user's question
 * @param {string} movieId - The movie to search
 * @param {number|null} timestamp - Optional timestamp hint (seconds); ignored when alreadyWatched=true
 * @param {Array} conversationHistory - [{role, content}, ...] prior turns
 * @param {boolean} alreadyWatched - If true, disables all spoiler protection (full knowledge mode)
 * @param {object} callbacks - { onSources, onToken, onDone, onError }
 *   - onSources(sources, intent): called once with retrieved chunks
 *   - onToken(content): called for each streamed token
 *   - onDone(metadata): called when streaming completes
 *   - onError(message): called on error
 * @returns {AbortController} - call .abort() to cancel the stream
 */
export function askOracleStream(query, movieId, timestamp, conversationHistory = [], alreadyWatched = false, callbacks = {}) {
    const { onSources, onToken, onDone, onError } = callbacks

    const controller = new AbortController()

    const payload = {
        query: query.trim(),
        movie_id: movieId || undefined,
        timestamp: timestamp ? parseFloat(timestamp) : undefined,
        top_k: 5,
        conversation_history: conversationHistory,
        already_watched: alreadyWatched || false,
    }

    ;(async () => {
        let doneSeen = false
        let errorSeen = false
        let sawSources = false
        let sawToken = false

        const handleEvent = (rawEvent) => {
            const line = rawEvent.trim()
            if (!line.startsWith('data:')) return
            try {
                const data = JSON.parse(line.slice(5).trim())
                if (data.type === 'sources') {
                    sawSources = true
                    onSources?.(data.sources, data.intent)
                } else if (data.type === 'token') {
                    sawToken = true
                    onToken?.(data.content)
                } else if (data.type === 'done') {
                    doneSeen = true
                    onDone?.(data)
                } else if (data.type === 'error') {
                    errorSeen = true
                    onError?.(data.message)
                }
            } catch (_) {
                // skip malformed SSE lines
            }
        }

        try {
            const res = await fetch(`${API_URL}/oracle/ask/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal,
            })

            if (!res.ok) {
                const text = await res.text()
                onError?.(text || `HTTP ${res.status}`)
                return
            }

            if (!res.body) {
                onError?.('Oracle stream is unavailable right now.')
                return
            }

            const reader = res.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })

                // SSE format: each event is "data: {...}\n\n"
                const events = buffer.split('\n\n')
                buffer = events.pop() // keep incomplete last chunk

                for (const event of events) {
                    handleEvent(event)
                }
            }

            // Parse any trailing event that may not end with a double newline.
            if (buffer.trim()) {
                handleEvent(buffer)
            }

            // Some network/proxy paths close the stream without forwarding the final
            // `done` event. Treat a clean EOF as completion if we already streamed data.
            if (!doneSeen && !errorSeen && (sawSources || sawToken)) {
                onDone?.({ type: 'done', model_used: 'unknown', query_time_ms: null, inferred: true })
            } else if (!doneSeen && !errorSeen) {
                onError?.('Oracle stream ended before any response data was received.')
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                onError?.(err.message || 'Stream failed')
            }
        }
    })()

    return controller // caller can call controller.abort() to cancel
}

export async function getOracleSuggestions(movieId) {
    try {
        const res = await axios.get(`${API_URL}/oracle/suggestions/${encodeURIComponent(movieId)}`)
        return res.data.suggestions || []
    } catch (_) {
        return []
    }
}

export async function analyzeSentiment(text) {
    const res = await axios.post(`${API_URL}/sentiment/analyze`, {
        text: text.trim(),
        include_explanation: true,
    })
    return res.data
}

export async function predictBinge(userId, watchHistory) {
    const res = await axios.post(`${API_URL}/binge/predict`, {
        user_id: userId,
        watch_history: watchHistory,
        current_hour: new Date().getHours(),
    })
    return res.data
}

export async function getModelInfo() {
    const res = await axios.get(`${API_URL}/model-info`)
    return res.data
}

export async function getMovieImages(tmdbId, title, year) {
    const normalizedId = Number(tmdbId)
    if (!Number.isFinite(normalizedId) || normalizedId <= 0) {
        return { ...DEFAULT_IMAGE_PAYLOAD }
    }

    if (movieImageCache.has(normalizedId)) {
        return movieImageCache.get(normalizedId)
    }

    if (Date.now() < tmdbCircuitOpenUntil) {
        const fallbackPoster = await fetchOmdbPoster(title, year)
        const fallback = normalizeImagePayload({
            ...DEFAULT_IMAGE_PAYLOAD,
            poster_path: fallbackPoster,
            posters: fallbackPoster ? [fallbackPoster] : [],
            tmdb_available: false,
            circuit_open: true,
            provider: fallbackPoster ? 'omdb' : 'none',
        })
        movieImageCache.set(normalizedId, fallback)
        return fallback
    }

    if (movieImageInFlight.has(normalizedId)) {
        return movieImageInFlight.get(normalizedId)
    }

    const req = axios
        .get(`${API_URL}/movies/${normalizedId}/images`, { timeout: 8000 })
        .then((res) => {
            const data = normalizeImagePayload(res.data || DEFAULT_IMAGE_PAYLOAD)
            if (data?.tmdb_available === false || !data?.poster_path) {
                tmdbCircuitOpenUntil = Date.now() + TMDB_CIRCUIT_COOLDOWN_MS
            } else {
                tmdbCircuitOpenUntil = 0
            }
            return data
        })
        .then(async (data) => {
            if (data?.posters?.length > 0) {
                const payload = normalizeImagePayload({ ...data, provider: 'tmdb' })
                movieImageCache.set(normalizedId, payload)
                return payload
            }

            const fallbackPoster = await fetchOmdbPoster(title, year)
            const payload = normalizeImagePayload({
                ...data,
                poster_path: fallbackPoster || data?.poster_path || null,
                posters: dedupeUrls([fallbackPoster, ...(data?.posters || [])]),
                provider: fallbackPoster ? 'omdb' : (data?.provider || 'none'),
            })
            movieImageCache.set(normalizedId, payload)
            return payload
        })
        .catch((err) => {
            tmdbCircuitOpenUntil = Date.now() + TMDB_CIRCUIT_COOLDOWN_MS
            return fetchOmdbPoster(title, year).then((fallbackPoster) => {
                const fallback = normalizeImagePayload({
                    ...DEFAULT_IMAGE_PAYLOAD,
                    poster_path: fallbackPoster,
                    posters: fallbackPoster ? [fallbackPoster] : [],
                    tmdb_available: false,
                    circuit_open: true,
                    provider: fallbackPoster ? 'omdb' : 'none',
                })
                movieImageCache.set(normalizedId, fallback)
                return fallback
            })
        })
        .finally(() => {
            movieImageInFlight.delete(normalizedId)
        })

    movieImageInFlight.set(normalizedId, req)
    return req
}

export async function prefetchMovieImages(items = []) {
    const normalized = (items || [])
        .map((item) => (typeof item === 'object' ? item : { tmdb_id: item }))
        .map((m) => ({
            tmdb_id: Number(m.tmdb_id),
            title: m.title,
            year: m.year,
        }))
        .filter((m) => Number.isFinite(m.tmdb_id) && m.tmdb_id > 0)

    const unique = []
    const seen = new Set()
    for (const m of normalized) {
        if (seen.has(m.tmdb_id)) continue
        seen.add(m.tmdb_id)
        unique.push(m)
    }

    if (unique.length === 0) return
    await Promise.all(unique.map((m) => getMovieImages(m.tmdb_id, m.title, m.year).catch(() => null)))
}
