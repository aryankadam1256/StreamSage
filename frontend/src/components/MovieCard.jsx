import { motion } from 'framer-motion'
import { Star, User, Tag, ChevronRight } from 'lucide-react'

export default function MovieCard({ movie, index = 0, onClick }) {
    const genre = movie.genres || movie.genre
    const rating = movie.rating ? parseFloat(movie.rating).toFixed(1) : null

    return (
        <motion.article
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, duration: 0.3, ease: 'easeOut' }}
            onClick={() => onClick?.(movie)}
            className="group bg-brand-surface border border-brand-border-subtle rounded-xl p-5
                       hover:bg-brand-card hover:border-brand-gold/15
                       cursor-pointer transition-all duration-200 shadow-card hover:shadow-card-hover"
        >
            <div className="flex items-start justify-between gap-3 mb-3">
                <div className="min-w-0">
                    <h3 className="font-semibold text-text-warm leading-tight group-hover:text-brand-gold
                                   transition-colors duration-150 text-base">
                        {movie.title}
                    </h3>
                    {movie.year && (
                        <span className="text-xs text-text-dim mt-0.5 block">{movie.year}</span>
                    )}
                </div>
                {rating && (
                    <div className="flex items-center gap-1 shrink-0 badge-gold">
                        <Star size={10} className="fill-amber-400 text-amber-400" />
                        <span>{rating}</span>
                    </div>
                )}
            </div>

            {(genre || movie.director) && (
                <div className="flex flex-wrap gap-1.5 mb-3">
                    {genre && (
                        <span className="badge-neutral flex items-center gap-1">
                            <Tag size={9} /> {genre}
                        </span>
                    )}
                    {movie.director && (
                        <span className="badge-neutral flex items-center gap-1">
                            <User size={9} /> {movie.director}
                        </span>
                    )}
                </div>
            )}

            {movie.description && (
                <p className="text-sm text-text-muted leading-relaxed line-clamp-3">{movie.description}</p>
            )}

            <div className="mt-4 flex items-center justify-end gap-1 text-xs text-text-dim
                            group-hover:text-brand-gold transition-colors duration-150">
                <span>Explore</span>
                <ChevronRight size={13} />
            </div>
        </motion.article>
    )
}
