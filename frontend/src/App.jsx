/**
 * =============================================================================
 * StreamSage Frontend - Main Application
 * =============================================================================
 * 
 * 🎓 CONCEPT: Component-Based Architecture
 * 
 * React's core principle is composability - building complex UIs from
 * small, reusable components.
 * 
 * Why Component Architecture?
 * 
 * 1. REUSABILITY: Write once, use everywhere
 * 2. SEPARATION OF CONCERNS: Each component has one job
 * 3. TESTABILITY: Test components in isolation
 * 4. MAINTAINABILITY: Easier to debug and update
 * 
 * Our component hierarchy:
 * 
 * App
 * ├── Header (Navigation)
 * ├── VibeBar (Sentiment Visualization)
 * ├── OracleChat (RAG Q&A Interface)
 * └── BingeGauge (Watch Prediction Display)
 * 
 * =============================================================================
 */

import { useState } from 'react'
import { motion } from 'framer-motion'
import Header from './components/Header'
import VibeBar from './components/VibeBar'
import OracleChat from './components/OracleChat'
import BingeGauge from './components/BingeGauge'

function App() {
    const [activeTab, setActiveTab] = useState('oracle')

    return (
        <div className="min-h-screen bg-cyber-bg">
            {/* Animated background grid */}
            <div className="fixed inset-0 opacity-10">
                <div className="absolute inset-0" style={{
                    backgroundImage: `
            linear-gradient(to right, #00f0ff 1px, transparent 1px),
            linear-gradient(to bottom, #00f0ff 1px, transparent 1px)
          `,
                    backgroundSize: '50px 50px',
                }} />
            </div>

            {/* Content */}
            <div className="relative z-10">
                <Header />

                {/* Hero Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    className="container mx-auto px-4 py-12"
                >
                    <div className="text-center mb-12">
                        <h1 className="text-6xl font-bold mb-4">
                            <span className="text-glow">StreamSage</span>
                        </h1>
                        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                            Intelligent movie analysis powered by{' '}
                            <span className="text-cyber-accent">RAG</span>,{' '}
                            <span className="text-cyber-purple">LSTM</span>, and{' '}
                            <span className="text-cyber-pink">BERT</span>
                        </p>
                    </div>

                    {/* Tab Navigation */}
                    <div className="flex justify-center gap-4 mb-8">
                        {[
                            { id: 'oracle', label: '🔮 Oracle Chat', desc: 'Ask about movie scenes' },
                            { id: 'vibe', label: '💬 Sentiment Vibe', desc: 'Analyze reviews' },
                            { id: 'binge', label: '📊 Binge Gauge', desc: 'Watch predictions' },
                        ].map((tab) => (
                            <motion.button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`px-6 py-3 rounded-lg transition-all ${activeTab === tab.id
                                        ? 'bg-gradient-to-r from-cyber-purple to-cyber-pink text-white shadow-lg'
                                        : 'glass-morphism text-gray-300 hover:bg-white/10'
                                    }`}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <div className="font-semibold">{tab.label}</div>
                                <div className="text-xs opacity-70">{tab.desc}</div>
                            </motion.button>
                        ))}
                    </div>

                    {/* Component Display */}
                    <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                        className="max-w-6xl mx-auto"
                    >
                        {activeTab === 'oracle' && <OracleChat />}
                        {activeTab === 'vibe' && <VibeBar />}
                        {activeTab === 'binge' && <BingeGauge />}
                    </motion.div>
                </motion.div>

                {/* Footer */}
                <footer className="container mx-auto px-4 py-8 text-center text-gray-500 text-sm">
                    <p>Built with 💜 for learning AI/ML through practical application</p>
                    <p className="mt-2">
                        Oracle (RAG) · Binge Predictor (LSTM) · Sentiment Engine (BERT)
                    </p>
                </footer>
            </div>
        </div>
    )
}

export default App
