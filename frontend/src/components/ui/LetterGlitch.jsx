import { useRef, useEffect } from 'react'

export default function LetterGlitch({
    glitchColors = ['#2b4539', '#61dca3', '#61b3dc'],
    className = '',
    glitchSpeed = 50,
    centerVignette = false,
    outerVignette = true,
    smooth = true,
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$&*()-_+=/[]{};:<>.,0123456789',
}) {
    const canvasRef = useRef(null)
    const animationRef = useRef(null)
    const letters = useRef([])
    const grid = useRef({ columns: 0, rows: 0 })
    const context = useRef(null)
    const lastGlitchTime = useRef(Date.now())
    const lettersAndSymbols = Array.from(characters)
    const fontSize = 16
    const charWidth = 10
    const charHeight = 20

    const getRandomChar = () => lettersAndSymbols[Math.floor(Math.random() * lettersAndSymbols.length)]
    const getRandomColor = () => glitchColors[Math.floor(Math.random() * glitchColors.length)]

    const hexToRgb = hex => {
        hex = hex.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i, (_, r, g, b) => r + r + g + g + b + b)
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
        return result ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) } : null
    }

    const interpolateColor = (start, end, factor) => {
        const r = Math.round(start.r + (end.r - start.r) * factor)
        const g = Math.round(start.g + (end.g - start.g) * factor)
        const b = Math.round(start.b + (end.b - start.b) * factor)
        return `rgb(${r}, ${g}, ${b})`
    }

    const calculateGrid = (w, h) => ({ columns: Math.ceil(w / charWidth), rows: Math.ceil(h / charHeight) })

    const initializeLetters = (columns, rows) => {
        grid.current = { columns, rows }
        letters.current = Array.from({ length: columns * rows }, () => ({
            char: getRandomChar(), color: getRandomColor(), targetColor: getRandomColor(), colorProgress: 1,
        }))
    }

    const drawLetters = () => {
        if (!context.current || letters.current.length === 0) return
        const ctx = context.current
        const { width, height } = canvasRef.current.getBoundingClientRect()
        ctx.clearRect(0, 0, width, height)
        ctx.font = `${fontSize}px monospace`
        ctx.textBaseline = 'top'
        letters.current.forEach((letter, index) => {
            const x = (index % grid.current.columns) * charWidth
            const y = Math.floor(index / grid.current.columns) * charHeight
            ctx.fillStyle = letter.color
            ctx.fillText(letter.char, x, y)
        })
    }

    const updateLetters = () => {
        if (!letters.current?.length) return
        const count = Math.max(1, Math.floor(letters.current.length * 0.05))
        for (let i = 0; i < count; i++) {
            const idx = Math.floor(Math.random() * letters.current.length)
            if (!letters.current[idx]) continue
            letters.current[idx].char = getRandomChar()
            letters.current[idx].targetColor = getRandomColor()
            if (!smooth) {
                letters.current[idx].color = letters.current[idx].targetColor
                letters.current[idx].colorProgress = 1
            } else {
                letters.current[idx].colorProgress = 0
            }
        }
    }

    const handleSmoothTransitions = () => {
        let needsRedraw = false
        letters.current.forEach(letter => {
            if (letter.colorProgress < 1) {
                letter.colorProgress = Math.min(letter.colorProgress + 0.05, 1)
                const s = hexToRgb(letter.color)
                const e = hexToRgb(letter.targetColor)
                if (s && e) { letter.color = interpolateColor(s, e, letter.colorProgress); needsRedraw = true }
            }
        })
        if (needsRedraw) drawLetters()
    }

    const resizeCanvas = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        const parent = canvas.parentElement
        if (!parent) return
        const dpr = window.devicePixelRatio || 1
        const rect = parent.getBoundingClientRect()
        canvas.width = rect.width * dpr
        canvas.height = rect.height * dpr
        canvas.style.width = `${rect.width}px`
        canvas.style.height = `${rect.height}px`
        if (context.current) context.current.setTransform(dpr, 0, 0, dpr, 0, 0)
        const { columns, rows } = calculateGrid(rect.width, rect.height)
        initializeLetters(columns, rows)
        drawLetters()
    }

    const animate = () => {
        const now = Date.now()
        if (now - lastGlitchTime.current >= glitchSpeed) {
            updateLetters()
            drawLetters()
            lastGlitchTime.current = now
        }
        if (smooth) handleSmoothTransitions()
        animationRef.current = requestAnimationFrame(animate)
    }

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        context.current = canvas.getContext('2d')
        resizeCanvas()
        animate()
        let resizeTimeout
        const handleResize = () => {
            clearTimeout(resizeTimeout)
            resizeTimeout = setTimeout(() => {
                cancelAnimationFrame(animationRef.current)
                resizeCanvas()
                animate()
            }, 100)
        }
        window.addEventListener('resize', handleResize)
        return () => { cancelAnimationFrame(animationRef.current); window.removeEventListener('resize', handleResize) }
    }, [glitchSpeed, smooth])

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%', backgroundColor: '#000', overflow: 'hidden' }} className={className}>
            <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
            {outerVignette && <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'radial-gradient(circle, rgba(0,0,0,0) 60%, rgba(0,0,0,1) 100%)' }} />}
            {centerVignette && <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: 'radial-gradient(circle, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 60%)' }} />}
        </div>
    )
}
