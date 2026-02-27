import { useRef, useState, Children, cloneElement } from 'react'
import { motion, useSpring, useTransform } from 'framer-motion'
import './Dock.css'

function DockItem({ children, className = '', onClick, mouseX, spring }) {
    const ref = useRef(null)
    const distance = useTransform(mouseX, (val) => {
        const el = ref.current
        if (!el || val === -999) return 150
        const rect = el.getBoundingClientRect()
        return Math.abs(val - (rect.left + rect.width / 2))
    })
    const scale = useTransform(distance, [0, 100, 200], [1.35, 1.1, 1])
    const smoothScale = useSpring(scale, spring)

    return (
        <motion.button
            ref={ref}
            className={`dock-item ${className}`}
            style={{ scale: smoothScale }}
            onClick={onClick}
            whileTap={{ scale: 0.9 }}
        >
            {children}
        </motion.button>
    )
}

export default function Dock({
    items = [],
    className = '',
    spring = { mass: 0.15, stiffness: 170, damping: 12 },
}) {
    const mouseX = useSpring(-999, spring)

    return (
        <motion.div
            className={`dock-container ${className}`}
            onMouseMove={(e) => mouseX.set(e.clientX)}
            onMouseLeave={() => mouseX.set(-999)}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.2 }}
        >
            {items.map((item, idx) => (
                <DockItem
                    key={idx}
                    mouseX={mouseX}
                    spring={spring}
                    onClick={item.onClick}
                    className={item.active ? 'dock-item-active' : ''}
                >
                    <div className="dock-icon">{item.icon}</div>
                    <span className="dock-label">{item.label}</span>
                    {item.active && <div className="dock-indicator" />}
                </DockItem>
            ))}
        </motion.div>
    )
}
