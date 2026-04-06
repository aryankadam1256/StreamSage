import { motion } from 'framer-motion'
import { Star, User, Tag, ChevronRight, Sparkles } from 'lucide-react'
import { useMovieImages } from '../hooks/useMovieImages'

export default function MovieCard({ movie, index = 0, onClick }) {
    const genre = movie.genres || movie.genre
    const rating = movie.rating ? parseFloat(movie.rating).toFixed(1) : null
    
    // Alternate sides for AI take based on index
    const isEven = index % 2 === 0;
    const images = useMovieImages(movie.tmdb_id, movie.title, movie.year);
    const fallbackTitle = (movie.title || 'Untitled').toUpperCase();

    return (
        <motion.article
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, duration: 0.3, ease: 'easeOut' }}
            onClick={() => onClick?.(movie)}
            className="group bg-brand-surface border border-brand-border-subtle rounded-xl p-0 
                       hover:bg-brand-card hover:border-brand-gold/15 cursor-pointer transition-all duration-200 
                       shadow-card hover:shadow-card-hover flex flex-col md:flex-row overflow-hidden"
        >
            {/* Poster Side */}
            <div className={`w-full md:w-32 lg:w-48 shrink-0 bg-brand-surface border-b md:border-b-0 md:border-r border-brand-border-subtle relative overflow-hidden flex items-center justify-center text-text-muted text-sm ${!isEven ? 'md:order-1' : ''}`}>
                {images.poster ? (
                    <img 
                      src={images.poster} 
                      alt={movie.title} 
                      className="w-full h-full object-cover aspect-[2/3] md:aspect-auto" 
                      loading="lazy"
                      decoding="async"
                      referrerPolicy="no-referrer"
                    />
                ) : images.loading ? (
                    <div className="w-full h-full min-h-[240px] md:min-h-0 animate-pulse bg-gradient-to-br from-brand-card to-brand-surface" />
                ) : (
                    <div className="w-full h-full min-h-[240px] md:min-h-0 bg-gradient-to-br from-brand-surface via-brand-card to-brand-surface p-4 flex flex-col justify-end">
                        <div className="text-[10px] uppercase tracking-[0.2em] text-brand-gold/70 mb-2">StreamSage</div>
                        <div className="text-sm font-semibold leading-snug text-text-warm line-clamp-4">{fallbackTitle}</div>
                        {movie.year && <div className="text-xs text-text-dim mt-2">{movie.year}</div>}
                    </div>
                )}
            </div>

            {/* Movie Info Side */}
            <div className={`flex flex-col flex-1 p-5 ${!isEven ? 'md:order-2' : ''}`}>
                <div className="flex items-start justify-between gap-3 mb-3">
                    <div className="min-w-0">
                        <h3 className="font-semibold text-text-warm leading-tight group-hover:text-brand-gold 
                                       transition-colors duration-150 text-xl">
                            {movie.title}
                        </h3>
                        {movie.year && (
                            <span className="text-sm text-text-dim mt-1 block">{movie.year}</span>
                        )}
                    </div>
                    {rating && (
                        <div className="flex items-center gap-1 shrink-0 badge-gold">
                            <Star size={12} className="fill-amber-400 text-amber-400" />
                            <span className="font-medium text-sm">{rating}</span>
                        </div>
                    )}
                </div>

                {(genre || movie.director) && (
                    <div className="flex flex-wrap gap-2 mb-4">
                        {genre && (
                            <span className="badge-neutral flex items-center gap-1">
                                <Tag size={12} /> {genre}
                            </span>
                        )}
                        {movie.director && (
                            <span className="badge-neutral flex items-center gap-1">
                                <User size={12} /> {movie.director}
                            </span>
                        )}
                    </div>
                )}

                {movie.description && (
                    <p className="text-sm text-text-muted leading-relaxed mb-4 line-clamp-3 md:line-clamp-none">{movie.description}</p>
                )}

                <div className="mt-auto flex items-center justify-end gap-1 text-sm text-text-dim 
                                group-hover:text-brand-gold transition-colors duration-150 pt-3">
                    <span>Explore Context</span>
                    <ChevronRight size={16} />
                </div>
            </div>

            {/* AI Explanation Side */}
            {movie.recommendation_reason && (
                <div className={`flex flex-col w-full md:w-1/3 shrink-0 p-6 bg-purple-900/10 border-purple-500/20 
                                ${isEven ? 'border-t md:border-t-0 md:border-l' : 'border-t md:border-t-0 md:border-r'} 
                                ${!isEven ? 'md:order-1' : ''} flex justify-center`}>
                    <div className="flex flex-col gap-3">
                        <h4 className="flex items-center gap-2 text-sm font-bold text-purple-400 uppercase tracking-wider">
                            <Sparkles size={16} className="text-purple-400" />
                            AI's Take
                        </h4>
                        <p className="text-gray-300 text-sm md:text-base italic leading-relaxed">
                            "{movie.recommendation_reason.replace(/^🤖 AI's Take:\s*/, '').replace(/^💡 Why you should watch:\s*/, '').replace(/^✨ AI's Take:\s*/, '')}"
                        </p>
                    </div>
                </div>
            )}
        </motion.article>
    )
}
