import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

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
                    const line = event.trim()
                    if (!line.startsWith('data:')) continue
                    try {
                        const data = JSON.parse(line.slice(5).trim())
                        if (data.type === 'sources') {
                            onSources?.(data.sources, data.intent)
                        } else if (data.type === 'token') {
                            onToken?.(data.content)
                        } else if (data.type === 'done') {
                            onDone?.(data)
                        } else if (data.type === 'error') {
                            onError?.(data.message)
                        }
                    } catch (_) {
                        // skip malformed SSE lines
                    }
                }
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
