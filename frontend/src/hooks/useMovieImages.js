import { useState, useEffect } from 'react';
import { getMovieImages } from '../api';

const imageCache = new Map();

export function useMovieImages(tmdbId, title, year) {
  const [images, setImages] = useState({ poster: null, backdrop: null, posters: [], backdrops: [], loading: false, error: null });

  useEffect(() => {
    const normalizedId = Number(tmdbId);
    if (!Number.isFinite(normalizedId) || normalizedId <= 0) {
      setImages({ poster: null, backdrop: null, posters: [], backdrops: [], loading: false, error: null });
      return;
    }

    if (imageCache.has(normalizedId)) {
      setImages({ ...imageCache.get(normalizedId), loading: false, error: null });
      return;
    }

    let isMounted = true;
    setImages(prev => ({ ...prev, loading: true, error: null }));

    getMovieImages(normalizedId, title, year)
      .then(data => {
        const payload = {
          poster: data?.poster_path || null,
          backdrop: data?.backdrop_path || null,
          posters: data?.posters || (data?.poster_path ? [data.poster_path] : []),
          backdrops: data?.backdrops || (data?.backdrop_path ? [data.backdrop_path] : []),
        };
        imageCache.set(normalizedId, payload);
        if (isMounted) {
          setImages({ ...payload, loading: false, error: null });
        }
      })
      .catch(err => {
        if (isMounted) {
          setImages(prev => ({ ...prev, loading: false, error: err.message }));
        }
      });

    return () => { isMounted = false; };
  }, [tmdbId, title, year]);

  return images;
}
