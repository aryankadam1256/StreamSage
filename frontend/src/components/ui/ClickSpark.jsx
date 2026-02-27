import { useRef, useEffect, useCallback } from 'react'

export default function ClickSpark({
    sparkColor = '#d4a017',
    sparkCount = 10,
    sparkSize = 12,
    duration = 500,
    children,
}) {
    const containerRef = useRef(null)

    const createSpark = useCallback((x, y) => {
        const container = containerRef.current
        if (!container) return
        const rect = container.getBoundingClientRect()
        const localX = x - rect.left
        const localY = y - rect.top

        for (let i = 0; i < sparkCount; i++) {
            const spark = document.createElement('div')
            const angle = (360 / sparkCount) * i + (Math.random() * 20 - 10)
            const distance = 20 + Math.random() * 35
            const size = sparkSize * (0.4 + Math.random() * 0.6)
            const rad = (angle * Math.PI) / 180

            // Film-reel / star shapes
            const shapes = ['star', 'circle', 'diamond']
            const shape = shapes[Math.floor(Math.random() * shapes.length)]

            Object.assign(spark.style, {
                position: 'absolute',
                left: `${localX}px`,
                top: `${localY}px`,
                width: `${size}px`,
                height: `${size}px`,
                pointerEvents: 'none',
                zIndex: '9999',
                transition: `all ${duration}ms cubic-bezier(0.22, 1, 0.36, 1)`,
                opacity: '1',
            })

            if (shape === 'star') {
                spark.style.background = 'none'
                spark.style.clipPath = 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)'
                spark.style.backgroundColor = sparkColor
            } else if (shape === 'diamond') {
                spark.style.clipPath = 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)'
                spark.style.backgroundColor = sparkColor
            } else {
                spark.style.borderRadius = '50%'
                spark.style.backgroundColor = sparkColor
            }

            container.appendChild(spark)

            requestAnimationFrame(() => {
                spark.style.transform = `translate(${Math.cos(rad) * distance}px, ${Math.sin(rad) * distance}px) scale(0.2)`
                spark.style.opacity = '0'
            })

            setTimeout(() => {
                if (spark.parentNode) spark.parentNode.removeChild(spark)
            }, duration + 50)
        }
    }, [sparkColor, sparkCount, sparkSize, duration])

    useEffect(() => {
        const container = containerRef.current
        if (!container) return
        const handler = (e) => createSpark(e.clientX, e.clientY)
        container.addEventListener('click', handler)
        return () => container.removeEventListener('click', handler)
    }, [createSpark])

    return (
        <div ref={containerRef} style={{ position: 'relative', overflow: 'hidden' }}>
            {children}
        </div>
    )
}
