import { motion } from 'framer-motion'

/**
 * Header Component - Navigation Bar
 * 
 * 🎓 Simple presentational component with no state
 */
export default function Header() {
    return (
        <header className="border-b border-cyber-accent/20 backdrop-blur-md bg-cyber-bg/80 sticky top-0 z-50">
            <div className="container mx-auto px-4 py-4">
                <div className="flex justify-between items-center">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center gap-3"
                    >
                        <div className="text-3xl">🎬</div>
                        <div>
                            <h1 className="text-2xl font-bold text-glow">StreamSage</h1>
                            <p className="text-xs text-gray-400">AI Movie Intelligence</p>
                        </div>
                    </motion.div>

                    <motion.nav
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex gap-6"
                    >
                        <a href="#" className="text-gray-300 hover:text-cyber-accent transition-colors">
                            Dashboard
                        </a>
                        <a href="#" className="text-gray-300 hover:text-cyber-accent transition-colors">
                            Analytics
                        </a>
                        <a href="#" className="text-gray-300 hover:text-cyber-accent transition-colors">
                            Settings
                        </a>
                    </motion.nav>
                </div>
            </div>
        </header>
    )
}
