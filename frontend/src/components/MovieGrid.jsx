import MovieCard from './MovieCard'

export default function MovieGrid({ movies, onMovieClick }) {
    if (!movies || movies.length === 0) return null

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {movies.map((movie, idx) => (
                <MovieCard
                    key={`${movie.title}-${idx}`}
                    movie={movie}
                    index={idx}
                    onClick={onMovieClick}
                />
            ))}
        </div>
    )
}
