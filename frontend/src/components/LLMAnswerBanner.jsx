import { useState } from 'react'
import { motion } from 'framer-motion'
import { Bot, ChevronDown, ChevronUp, Database } from 'lucide-react'

export default function LLMAnswerBanner({ answer, modelUsed, retrievalCount }) {
    const [expanded, setExpanded] = useState(true)
    if (!answer) return null
    const backend = modelUsed?.match(/\((\w+)\)$/)?.[1] || modelUsed || ''

    return (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }} className="max-w-4xl mx-auto mb-6">
            <div className="bg-brand-surface border border-brand-border-subtle rounded-xl overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-brand-border-subtle">
                    <div className="flex items-center gap-3">
                        <div className="w-6 h-6 bg-brand-gold/10 rounded flex items-center justify-center">
                            <Bot size={13} className="text-brand-gold" />
                        </div>
                        <span className="text-sm font-medium text-text-warm">AI Recommendation</span>
                        <div className="flex items-center gap-3 text-xs text-text-dim">
                            {retrievalCount !== undefined && (
                                <span className="flex items-center gap-1"><Database size={10} />{retrievalCount} sources</span>
                            )}
                            {backend && <span className="badge-neutral">{backend}</span>}
                        </div>
                    </div>
                    <button onClick={() => setExpanded(!expanded)} className="btn-ghost text-xs">
                        {expanded ? <><ChevronUp size={14} />Collapse</> : <><ChevronDown size={14} />Expand</>}
                    </button>
                </div>
                {expanded && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="px-4 py-4">
                        <p className="text-text-muted text-sm leading-relaxed whitespace-pre-wrap">{answer}</p>
                    </motion.div>
                )}
            </div>
        </motion.div>
    )
}
