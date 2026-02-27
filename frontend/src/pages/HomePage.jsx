import { motion } from 'framer-motion'
import { Film } from 'lucide-react'
import SearchHero from '../components/SearchHero'
import LLMAnswerBanner from '../components/LLMAnswerBanner'
import MovieGrid from '../components/MovieGrid'

export default function HomePage({ searchState, onSearch, onMovieClick }) {
    const { results, llmAnswer, loading, error, metrics } = searchState
    const hasResults = results && results.length > 0

    return (
        <div>
            <SearchHero onSearch={onSearch} loading={loading} />

            <div className="max-w-7xl mx-auto px-4 sm:px-6">
                {error && (
                    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                        className="max-w-2xl mx-auto mb-6 px-4 py-3 rounded-xl bg-brand-crimson/10
                                   border border-brand-crimson/20 text-brand-crimson">
                        <p className="font-medium text-sm">Something went wrong</p>
                        <p className="text-xs mt-0.5 opacity-70">{error}</p>
                    </motion.div>
                )}

                {llmAnswer && (
                    <LLMAnswerBanner answer={llmAnswer} modelUsed={metrics?.model_used}
                        retrievalCount={metrics?.retrieval_count} />
                )}

                {hasResults && (
                    <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }} className="max-w-5xl mx-auto pb-16">
                        <div className="flex items-center gap-2 mb-4">
                            <Film size={15} className="text-text-muted" />
                            <span className="text-sm text-text-muted">
                                {results.length} film{results.length !== 1 ? 's' : ''} found
                            </span>
                        </div>
                        <MovieGrid movies={results} onMovieClick={onMovieClick} />
                    </motion.section>
                )}

                {!results && !loading && !error && (
                    <div className="max-w-lg mx-auto text-center pb-16">
                        <div className="inline-flex flex-col items-center gap-4 px-6 py-8
                                        bg-brand-surface border border-brand-border-subtle rounded-xl">
                            <div className="w-10 h-10 bg-brand-card rounded-full flex items-center justify-center">
                                <Film size={18} className="text-text-muted" />
                            </div>
                            <div>
                                <p className="text-sm font-medium text-text-warm mb-1">How it works</p>
                                <p className="text-xs text-text-muted leading-relaxed">
                                    Your query is embedded as a vector. ChromaDB retrieves semantically
                                    similar films, and a fine-tuned Llama&nbsp;3&nbsp;8B generates
                                    personalized recommendations from those results.
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
