import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'

export default function BlurText({
    text = '',
    delay = 0.08,
    className = '',
}) {
    const words = text.split(' ')
    return (
        <span className={className}>
            {words.map((word, i) => (
                <motion.span
                    key={i}
                    className="inline-block mr-[0.3em]"
                    initial={{ filter: 'blur(12px)', opacity: 0, y: 8 }}
                    animate={{ filter: 'blur(0px)', opacity: 1, y: 0 }}
                    transition={{
                        duration: 0.5,
                        delay: i * delay,
                        ease: [0.25, 0.1, 0.25, 1],
                    }}
                >
                    {word}
                </motion.span>
            ))}
        </span>
    )
}
