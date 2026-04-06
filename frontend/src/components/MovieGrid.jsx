import MovieCard from './MovieCard'

export default function MovieGrid({ movies, onMovieClick }) {
    if (!movies || movies.length === 0) return null

    return (
        <div className="flex flex-col gap-6">
            {movies.map((movie, idx) => (
                <MovieCard
                    key={movie.tmdb_id || `${movie.title}-${idx}`}
                    movie={movie}
                    index={idx}
                    onClick={onMovieClick}
                />
            ))}
        </div>
    )
}
