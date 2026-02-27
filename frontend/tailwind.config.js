/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    bg: '#080810',
                    surface: '#0e0e18',
                    card: '#161622',
                    'card-hover': '#1e1e2e',
                    gold: '#d4a017',
                    'gold-hover': '#e8b420',
                    'gold-muted': '#a07c12',
                    crimson: '#c0103a',
                    border: '#202030',
                    'border-subtle': '#14141e',
                },
                text: {
                    warm: '#e8e0d0',
                    muted: '#6a6070',
                    dim: '#3a3545',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['Fira Code', 'monospace'],
            },
            boxShadow: {
                'card': '0 2px 12px rgba(0,0,0,0.7)',
                'card-hover': '0 8px 32px rgba(0,0,0,0.9)',
                'gold': '0 4px 24px rgba(212,160,23,0.15)',
                'gold-lg': '0 8px 40px rgba(212,160,23,0.2)',
            },
        },
    },
    plugins: [],
}
