import { useState } from 'react'
import Header from './components/Header'
import HomePage from './pages/HomePage'
import MovieDetailPage from './pages/MovieDetailPage'
import ClickSpark from './components/ui/ClickSpark'
import { discoverMovies } from './api'

function App() {
    const [currentPage, setCurrentPage] = useState('home')
    const [selectedMovie, setSelectedMovie] = useState(null)
    const [searchState, setSearchState] = useState({
        query: '', results: null, llmAnswer: null, loading: false, error: null, metrics: null,
    })

    const handleSearch = async (query, filters = {}) => {
        setSearchState(prev => ({ ...prev, query, loading: true, error: null }))
        try {
            const data = await discoverMovies(query, filters)
            setSearchState(prev => ({
                ...prev, loading: false,
                results: data.recommended_movies || [],
                llmAnswer: data.answer || data.response || null,
                metrics: { model_used: data.model_used, retrieval_count: data.retrieval_count },
            }))
        } catch (err) {
            setSearchState(prev => ({
                ...prev, loading: false,
                error: err.response?.data?.detail || err.message || 'Failed to get recommendations',
            }))
        }
    }

    const handleMovieClick = (movie) => { setSelectedMovie(movie); setCurrentPage('detail'); window.scrollTo(0, 0) }
    const handleNavigateHome = () => { setCurrentPage('home'); setSelectedMovie(null) }

    return (
        <ClickSpark sparkColor="#d4a017" sparkCount={8} sparkSize={10} duration={600}>
            <div className="min-h-screen bg-brand-bg">
                <Header currentPage={currentPage} onNavigateHome={handleNavigateHome} />

                <main>
                    {currentPage === 'home' && (
                        <HomePage searchState={searchState} onSearch={handleSearch} onMovieClick={handleMovieClick} />
                    )}
                    {currentPage === 'detail' && selectedMovie && (
                        <MovieDetailPage movie={selectedMovie} onBack={handleNavigateHome} />
                    )}
                </main>

                <footer className="border-t border-brand-border-subtle mt-8 py-8 px-4">
                    <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-text-dim">
                        <span className="font-medium text-text-muted">StreamSage</span>
                        <span>Oracle RAG · Binge LSTM · Sentiment BERT · Discovery Llama 3</span>
                    </div>
                </footer>
            </div>
        </ClickSpark>
    )
}

export default App
