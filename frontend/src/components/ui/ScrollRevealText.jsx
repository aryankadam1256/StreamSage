import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'

export default function ScrollRevealText({ text, className = '', delay = 0 }) {
    const ref = useRef(null)
    const isInView = useInView(ref, { once: true, margin: '-20px' })
    const words = text.split(' ')

    return (
        <span ref={ref} className={className}>
            {words.map((word, i) => (
                <span key={i} className="inline-block overflow-hidden mr-[0.3em]">
                    <motion.span
                        className="inline-block"
                        initial={{ y: '100%', opacity: 0 }}
                        animate={isInView ? { y: '0%', opacity: 1 } : {}}
                        transition={{
                            duration: 0.4,
                            delay: delay + i * 0.03,
                            ease: [0.25, 0.4, 0.25, 1],
                        }}
                    >
                        {word}
                    </motion.span>
                </span>
            ))}
        </span>
    )
}
