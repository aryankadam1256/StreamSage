import { useState, useEffect, useRef } from 'react'
import { motion, useInView, AnimatePresence } from 'framer-motion'

/* ── helpers ──────────────────────────────────────────────────────────────── */
const FadeUp = ({ children, delay = 0, className = '' }) => {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })
  return (
    <motion.div ref={ref}
      initial={{ opacity: 0, y: 32 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}>
      {children}
    </motion.div>
  )
}

/* ── ticker logos (genres / moods) ───────────────────────────────────────── */
const tickers = [
  { name: 'Sci-Fi',      style: 'font-bold tracking-tight' },
  { name: 'Thriller',    style: 'font-semibold italic' },
  { name: 'Drama',       style: 'font-light tracking-widest uppercase text-sm' },
  { name: 'Action',      style: 'font-black tracking-tight' },
  { name: 'Horror',      style: 'font-semibold tracking-tight' },
  { name: 'Romance',     style: 'font-bold' },
  { name: 'Animation',   style: 'font-extrabold tracking-tight' },
  { name: 'Documentary', style: 'font-semibold' },
]

/* ── interactive feature list ────────────────────────────────────────────── */
const features = [
  {
    icon: '🔮',
    title: 'Ask the Oracle anything',
    desc: 'RAG-powered chat lets you ask deep questions about any movie — plot, cast, themes, dialogue — with timestamped source citations.',
    color: '#ede9fe',
    accent: '#7c3aed',
    preview: (
      <div className="space-y-3">
        {[
          { q: 'What is the hidden meaning behind the ending?', a: 'The ending is a metaphor for...' },
          { q: 'Who wrote the screenplay?', a: 'Christopher Nolan wrote...' },
          { q: 'What influenced the visual style?', a: 'Kubrick\'s 2001 was a key...' },
        ].map((item, i) => (
          <div key={i} className="bg-white rounded-xl p-3 shadow-sm border border-gray-100">
            <p className="text-xs font-semibold text-purple-600 mb-1">You asked: {item.q}</p>
            <p className="text-xs text-gray-500 flex items-start gap-1">
              <span className="text-purple-400">🔮</span> {item.a}
            </p>
          </div>
        ))}
      </div>
    ),
  },
  {
    icon: '🎭',
    title: 'Vibe check any review',
    desc: 'BERT-powered sentiment analysis reads any text about a movie and instantly tells you if it\'s hype or hate — with confidence scores.',
    color: '#fce7f3',
    accent: '#db2777',
    preview: (
      <div className="space-y-3">
        <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
          <p className="text-xs text-gray-400 mb-1 font-medium">Review analyzed</p>
          <p className="text-xs text-gray-600 italic mb-3">"A masterpiece of modern cinema, deeply moving..."</p>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
              <motion.div className="h-full bg-green-400 rounded-full"
                initial={{ width: 0 }} whileInView={{ width: '92%' }} viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.3 }} />
            </div>
            <span className="text-sm font-bold text-green-600">92% 😍</span>
          </div>
        </div>
        <div className="bg-pink-50 rounded-xl p-3 border border-pink-100 text-xs text-pink-700 font-semibold">
          ✅ Positive · Confidence: 0.94 · BERT cuda
        </div>
      </div>
    ),
  },
  {
    icon: '📺',
    title: 'Will you binge it?',
    desc: 'Our LSTM model studies your watch history and predicts — before you press play — whether you\'ll finish the whole series.',
    color: '#fef9c3',
    accent: '#ca8a04',
    preview: (
      <div className="space-y-3">
        <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
          <p className="text-xs text-gray-400 mb-3 font-medium">Binge probability</p>
          <div className="relative h-28 flex items-end justify-center">
            <svg viewBox="0 0 120 70" className="w-full">
              <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="#f3f4f6" strokeWidth="10" strokeLinecap="round"/>
              <motion.path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="#f59e0b" strokeWidth="10"
                strokeLinecap="round" strokeDasharray="157"
                initial={{ strokeDashoffset: 157 }}
                whileInView={{ strokeDashoffset: 47 }}
                viewport={{ once: true }}
                transition={{ duration: 1, delay: 0.3, ease: 'easeOut' }} />
              <text x="60" y="58" textAnchor="middle" className="text-lg font-black" fontSize="18" fontWeight="900" fill="#111">70%</text>
            </svg>
          </div>
        </div>
        <div className="bg-amber-50 rounded-xl p-3 border border-amber-100 text-xs text-amber-800 font-semibold">
          🔥 High binge risk — clear your weekend.
        </div>
      </div>
    ),
  },
  {
    icon: '🧠',
    title: 'Llama 3 recommendations',
    desc: 'Fine-tuned Llama 3 understands mood, theme, and context — not just genre tags — to surface hidden gems you\'ll actually love.',
    color: '#dcfce7',
    accent: '#16a34a',
    preview: (
      <div className="space-y-2">
        {[
          { title: 'Annihilation', year: '2018', match: '97%' },
          { title: 'Ex Machina',   year: '2014', match: '94%' },
          { title: 'Arrival',      year: '2016', match: '91%' },
        ].map((m, i) => (
          <div key={i} className="flex items-center gap-3 bg-white rounded-xl p-3 shadow-sm border border-gray-100">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
              {m.title[0]}
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-gray-800">{m.title}</p>
              <p className="text-xs text-gray-400">{m.year}</p>
            </div>
            <span className="text-xs font-bold text-green-600 bg-green-50 px-2 py-0.5 rounded-full">{m.match}</span>
          </div>
        ))}
      </div>
    ),
  },
  {
    icon: '⚡',
    title: 'GPU-accelerated everything',
    desc: 'RTX 4080-powered inference across all four AI models. Streaming responses, real-time embeddings, zero lag.',
    color: '#e0f2fe',
    accent: '#0284c7',
    preview: (
      <div className="space-y-2">
        {[
          { label: 'Llama 3 (LoRA)',         val: 88, color: '#16a34a' },
          { label: 'BERT Sentiment',         val: 95, color: '#db2777' },
          { label: 'LSTM Binge',             val: 99, color: '#ca8a04' },
          { label: 'BGE Embeddings',         val: 92, color: '#7c3aed' },
        ].map((s, i) => (
          <div key={i} className="bg-white rounded-xl px-3 py-2 shadow-sm border border-gray-100">
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-medium text-gray-600">{s.label}</span>
              <span className="text-xs font-bold" style={{ color: s.color }}>{s.val}% GPU</span>
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <motion.div className="h-full rounded-full"
                style={{ backgroundColor: s.color }}
                initial={{ width: 0 }}
                whileInView={{ width: `${s.val}%` }}
                viewport={{ once: true }}
                transition={{ duration: 0.7, delay: i * 0.1 }} />
            </div>
          </div>
        ))}
      </div>
    ),
  },
]

/* ── bento items ─────────────────────────────────────────────────────────── */
const bentoItems = [
  { icon: '🧠', title: 'Llama 3 Discovery',     desc: 'Fine-tuned LLM that actually understands what you want to watch', wide: true },
  { icon: '🔮', title: 'Oracle RAG Q&A',         desc: 'Ask anything about any movie, grounded in real data' },
  { icon: '🎭', title: 'BERT Sentiment',         desc: 'Instant positive/negative vibe check on any review' },
  { icon: '📺', title: 'LSTM Binge Predictor',   desc: 'Predicts series completion based on your watch history' },
  { icon: '🗂️', title: 'ChromaDB Vector Search', desc: 'Semantic search across 6,000+ films' },
  { icon: '⚡', title: 'GPU Streaming',           desc: 'RTX 4080-powered real-time inference' },
  { icon: '🎬', title: '6,000+ Movies',          desc: 'Massive indexed catalogue with subtitles and metadata', wide: true },
  { icon: '🔍', title: 'Semantic Filters',       desc: 'Filter by mood, era, genre, director, and more' },
  { icon: '💬', title: 'Streaming Responses',    desc: 'Token-by-token SSE output — no waiting' },
  { icon: '📊', title: 'Model Metadata',         desc: 'See exactly which model answered and how confident it is' },
  { icon: '🌐', title: 'Microservice API',       desc: 'Gateway routing to 4 independent AI services', wide: true },
]

/* ── avatar gradients ────────────────────────────────────────────────────── */
const avatarGrads = [
  'from-violet-400 to-purple-600',
  'from-blue-400 to-cyan-600',
  'from-green-400 to-emerald-600',
  'from-amber-400 to-orange-500',
  'from-rose-400 to-red-600',
]

/* ── product mockup (StreamSage search UI) ───────────────────────────────── */
const ProductMockup = () => (
  <div className="w-full rounded-2xl overflow-hidden shadow-[0_32px_80px_-12px_rgba(0,0,0,0.18)] border border-gray-200">
    {/* window chrome */}
    <div className="h-9 bg-gray-100 border-b border-gray-200 flex items-center px-4 gap-1.5">
      <div className="w-3 h-3 rounded-full bg-red-400" />
      <div className="w-3 h-3 rounded-full bg-amber-400" />
      <div className="w-3 h-3 rounded-full bg-green-400" />
      <div className="flex-1 mx-4 h-5 bg-white rounded border border-gray-200 flex items-center justify-center">
        <span className="text-xs text-gray-400">streamsage.ai</span>
      </div>
    </div>

    {/* app body */}
    <div className="bg-[#080810] p-5">
      {/* search bar */}
      <div className="flex items-center gap-2 bg-[#161622] rounded-xl px-4 py-3 border border-[#202030] mb-5">
        <span className="text-yellow-500 text-sm">✦</span>
        <span className="text-[#e8e0d0] text-sm flex-1">mind-bending sci-fi with unreliable narrator</span>
        <motion.div animate={{ opacity: [1, 0, 1] }} transition={{ duration: 1.2, repeat: Infinity }}
          className="w-0.5 h-4 bg-yellow-500 rounded" />
      </div>

      {/* movie cards */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { title: 'Inception',       year: 2010, genre: 'Sci-Fi', rating: '8.8', color: 'from-blue-900 to-slate-900' },
          { title: 'Memento',         year: 2000, genre: 'Thriller', rating: '8.4', color: 'from-gray-900 to-zinc-900' },
          { title: 'Shutter Island',  year: 2010, genre: 'Mystery', rating: '8.1', color: 'from-slate-900 to-gray-800' },
        ].map((m, i) => (
          <motion.div key={i}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + i * 0.15, duration: 0.5 }}
            className={`rounded-xl bg-gradient-to-br ${m.color} p-3 border border-white/5 cursor-pointer hover:border-yellow-500/30 transition-colors`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-yellow-400 font-bold">★ {m.rating}</span>
              <span className="text-xs text-gray-500">{m.year}</span>
            </div>
            <p className="text-white text-sm font-bold leading-tight mb-1">{m.title}</p>
            <span className="inline-block text-xs text-gray-400 bg-white/5 px-2 py-0.5 rounded-full">{m.genre}</span>
          </motion.div>
        ))}
      </div>

      {/* AI answer strip */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}
        className="mt-4 bg-[#161622] rounded-xl p-3 border border-yellow-500/20">
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-4 h-4 rounded bg-yellow-500/20 flex items-center justify-center">
            <span className="text-yellow-400 text-xs">AI</span>
          </div>
          <span className="text-xs text-yellow-400 font-semibold">Llama 3 · 3 results</span>
        </div>
        <p className="text-xs text-gray-400 leading-relaxed">
          "These films share a fractured timeline and an unreliable protagonist — perfect for viewers who love piecing together the truth..."
        </p>
      </motion.div>
    </div>
  </div>
)

/* ── Oracle RAG mockup ───────────────────────────────────────────────────── */
const OracleMockup = () => (
  <div className="rounded-2xl overflow-hidden shadow-[0_24px_60px_-8px_rgba(0,0,0,0.13)] border border-gray-200 bg-white">
    <div className="h-8 bg-gray-50 border-b border-gray-200 flex items-center px-4 gap-1.5">
      <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
      <div className="w-2.5 h-2.5 rounded-full bg-amber-400" />
      <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
      <span className="ml-3 text-xs text-gray-400 font-medium">Oracle RAG · Inception</span>
    </div>
    <div className="p-5 space-y-3">
      {[
        {
          q: 'What does the ending spinning top mean?',
          a: 'The top is Cobb\'s totem — if it falls, he\'s in reality; if it spins forever, he\'s dreaming. The final cut leaves it ambiguous...',
          src: 'Subtitle 01:58:22 — 01:58:34',
          score: '0.94',
        },
        {
          q: 'How many dream levels are there?',
          a: 'The heist goes four levels deep: the city, the hotel, the snow fortress, and limbo — each runs 20× slower than the level above.',
          src: 'Subtitle 00:44:10 — 00:44:28',
          score: '0.91',
        },
      ].map((item, i) => (
        <motion.div key={i}
          initial={{ opacity: 0, x: -12 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.15, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          className="rounded-xl bg-gray-50 p-3 border border-gray-100">
          <p className="text-xs font-semibold text-purple-600 mb-1.5">Q: {item.q}</p>
          <p className="text-xs text-gray-600 leading-relaxed mb-2">{item.a}</p>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 bg-white border border-gray-200 px-2 py-0.5 rounded-full">{item.src}</span>
            <span className="text-xs text-purple-600 font-semibold ml-auto">Score {item.score}</span>
          </div>
        </motion.div>
      ))}
    </div>
  </div>
)

/* ══════════════════════════════════════════════════════════════════════════ */
export default function LandingPage({ onEnterApp }) {
  const [scrolled,      setScrolled]      = useState(false)
  const [activeFeature, setActiveFeature] = useState(0)
  const [mobileOpen,    setMobileOpen]    = useState(false)

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 12)
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [])

  useEffect(() => {
    const id = setInterval(() => setActiveFeature(p => (p + 1) % features.length), 4000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="landing-page bg-white text-gray-900 overflow-x-hidden">

      {/* ── NAVBAR ──────────────────────────────────────────────────────── */}
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-white/90 backdrop-blur-md shadow-sm border-b border-gray-100' : 'bg-white/80 backdrop-blur-sm'
      }`}>
        <div className="max-w-7xl mx-auto px-5 h-16 flex items-center gap-4">
          {/* logo */}
          <a href="#" className="flex items-center gap-2 font-extrabold text-xl tracking-tight text-gray-900 mr-2">
            <div className="w-7 h-7 rounded-lg bg-[#d4a017] flex items-center justify-center">
              <svg viewBox="0 0 24 24" fill="white" className="w-4 h-4">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
              </svg>
            </div>
            StreamSage
          </a>

          <nav className="hidden md:flex items-center gap-1 flex-1">
            {['Discover','Oracle RAG','Sentiment','Binge AI','API Docs'].map(item => (
              <a key={item} href="#"
                className="px-3.5 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-50 transition-all">
                {item}
              </a>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <a href="#" className="hidden md:flex px-4 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-50 rounded-lg transition-all">
              View docs
            </a>
            <button onClick={onEnterApp}
              className="px-4 py-2 text-sm font-semibold text-white bg-[#d4a017] hover:bg-[#b8880f] rounded-lg transition-all shadow-sm hover:shadow-md active:scale-95">
              Try for free
            </button>
            <button className="md:hidden p-2 flex flex-col gap-1" onClick={() => setMobileOpen(o => !o)}>
              <div className="w-5 h-0.5 bg-gray-700 rounded" />
              <div className="w-5 h-0.5 bg-gray-700 rounded" />
              <div className="w-5 h-0.5 bg-gray-700 rounded" />
            </button>
          </div>
        </div>

        <AnimatePresence>
          {mobileOpen && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25 }}
              className="md:hidden bg-white border-t border-gray-100 px-5 pb-4 overflow-hidden">
              {['Discover','Oracle RAG','Sentiment','Binge AI','API Docs','View docs'].map(item => (
                <a key={item} href="#" className="block py-2.5 text-sm font-medium text-gray-700 border-b border-gray-50 hover:text-gray-900">{item}</a>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* ── HERO ────────────────────────────────────────────────────────── */}
      <section className="pt-32 pb-0 px-5 text-center overflow-hidden">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-50 border border-amber-200 text-sm font-semibold text-amber-700 mb-8">
          <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
          Powered by Llama 3 · BERT · LSTM · ChromaDB
          <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd"/>
          </svg>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          {['Discover films you\'ll love,', 'powered by real AI'].map((line, li) => (
            <div key={li} className="overflow-hidden">
              <motion.h1
                initial={{ y: 80, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.7, delay: 0.15 + li * 0.12, ease: [0.22, 1, 0.36, 1] }}
                className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.08] text-gray-950">
                {line}
              </motion.h1>
            </div>
          ))}
        </div>

        <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-6 text-lg text-gray-500 font-medium max-w-xl mx-auto leading-relaxed">
          Smart recommendations from a <span className="text-gray-700">fine-tuned Llama 3</span>. Deep Q&amp;A via RAG. Sentiment analysis. Binge prediction. All in one place.
        </motion.p>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
          <button onClick={onEnterApp}
            className="px-7 py-3.5 text-base font-bold text-gray-900 bg-[#d4a017] hover:bg-[#b8880f] rounded-xl transition-all shadow-lg hover:shadow-xl hover:-translate-y-0.5 active:scale-95">
            Explore movies →
          </button>
          <button onClick={onEnterApp}
            className="px-7 py-3.5 text-base font-bold text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 hover:border-gray-300 rounded-xl transition-all hover:-translate-y-0.5">
            See how it works
          </button>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.65 }}
          className="mt-5 flex items-center justify-center gap-3">
          <div className="flex -space-x-2">
            {avatarGrads.map((cls, i) => (
              <div key={i} className={`w-8 h-8 rounded-full border-2 border-white bg-gradient-to-br ${cls}`} />
            ))}
          </div>
          <p className="text-sm text-gray-500 font-medium">
            <span className="text-gray-900 font-bold">6,000+</span> movies indexed &amp; ready to explore
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 48 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.55, ease: [0.22, 1, 0.36, 1] }}
          className="mt-12 px-0 sm:px-6 max-w-5xl mx-auto">
          <motion.div animate={{ y: [0, -8, 0] }} transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}>
            <ProductMockup />
          </motion.div>
        </motion.div>
      </section>

      {/* ── GENRE TICKER ────────────────────────────────────────────────── */}
      <section className="py-14 border-t border-b border-gray-100 mt-16 overflow-hidden bg-gray-50/60">
        <p className="text-center text-xs font-semibold uppercase tracking-widest text-gray-400 mb-8">
          Every genre, every mood, every era
        </p>
        <div className="relative flex overflow-hidden">
          <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-gray-50 to-transparent z-10 pointer-events-none" />
          <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-gray-50 to-transparent z-10 pointer-events-none" />
          <div className="logo-ticker flex gap-16 items-center whitespace-nowrap">
            {[...tickers, ...tickers].map((t, i) => (
              <span key={i} className={`text-xl text-gray-400 hover:text-gray-700 transition-colors cursor-default select-none ${t.style}`}>
                {t.name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── ORACLE RAG SECTION ──────────────────────────────────────────── */}
      <section className="py-28 px-5 overflow-hidden">
        {/* scrolling "ask" ticker */}
        <div className="relative overflow-hidden mb-14 py-2.5 bg-purple-50 border-y border-purple-100">
          <div className="ai-ticker flex gap-8 whitespace-nowrap text-purple-500 text-xs font-bold uppercase tracking-widest">
            {Array.from({ length: 24 }).map((_, i) => (
              <span key={i} className="flex items-center gap-2 select-none">
                <span className="w-1 h-1 rounded-full bg-purple-400 inline-block" />
                ask
              </span>
            ))}
          </div>
        </div>

        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <FadeUp>
              <span className="inline-block px-3 py-1 rounded-full bg-purple-50 border border-purple-200 text-xs font-bold uppercase tracking-widest text-purple-600 mb-5">
                Oracle RAG
              </span>
              <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-950 leading-[1.1] tracking-tight">
                Ask any question about{' '}
                <span className="relative inline-block">
                  <span className="relative z-10">any movie</span>
                  <span className="absolute inset-x-0 bottom-1 h-3 bg-purple-200 opacity-60 rounded" />
                </span>
              </h2>
              <p className="mt-5 text-lg text-gray-500 leading-relaxed">
                RAG-powered answers grounded in actual subtitle data. Timestamped source citations, relevance scores, and spoiler protection built in.
              </p>
            </FadeUp>
            <FadeUp delay={0.15}>
              <div className="mt-7 flex flex-wrap gap-2">
                {['🔮 RAG answers','📌 Source citations','⏱️ Timestamps','🛡️ Spoiler guard','🎬 6K+ films'].map(tag => (
                  <span key={tag} className="px-3 py-1.5 rounded-full bg-gray-100 text-sm font-semibold text-gray-700 hover:bg-purple-50 hover:text-purple-800 transition-colors cursor-default">
                    {tag}
                  </span>
                ))}
              </div>
            </FadeUp>
            <FadeUp delay={0.25}>
              <button onClick={onEnterApp}
                className="mt-8 px-6 py-3 text-sm font-bold text-gray-900 bg-[#d4a017] hover:bg-[#b8880f] rounded-xl transition-all shadow hover:shadow-lg hover:-translate-y-0.5 active:scale-95">
                Ask the Oracle →
              </button>
            </FadeUp>
          </div>
          <FadeUp delay={0.1}><OracleMockup /></FadeUp>
        </div>
      </section>

      {/* ── AI FEATURES SECTION ─────────────────────────────────────────── */}
      <section className="py-28 px-5 bg-gray-50/60">
        <div className="max-w-7xl mx-auto">
          <FadeUp className="text-center mb-16">
            <span className="text-3xl mb-3 block">🧠</span>
            <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-950 tracking-tight">
              Four AI models, one seamless experience
            </h2>
            <p className="mt-4 text-lg text-gray-500 max-w-lg mx-auto">
              Each model trained for a specific job — working together to give you the full picture on any movie.
            </p>
          </FadeUp>

          <div className="grid lg:grid-cols-2 gap-12 items-start">
            <div className="space-y-1">
              {features.map((f, i) => (
                <button key={i} onClick={() => setActiveFeature(i)}
                  className={`w-full text-left p-5 rounded-2xl transition-all duration-300 group ${
                    activeFeature === i ? 'bg-white shadow-lg border border-gray-200' : 'hover:bg-white/60'
                  }`}>
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center text-xl flex-shrink-0 transition-all"
                      style={{
                        background: activeFeature === i ? f.color : '#f3f4f6',
                        transform: activeFeature === i ? 'scale(1.1)' : 'scale(1)',
                        opacity: activeFeature === i ? 1 : 0.55,
                      }}>
                      {f.icon}
                    </div>
                    <div className="flex-1">
                      <p className={`font-bold transition-colors ${activeFeature === i ? 'text-gray-900' : 'text-gray-500'}`}>
                        {f.title}
                      </p>
                      <AnimatePresence>
                        {activeFeature === i && (
                          <motion.p initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3 }}
                            className="text-sm text-gray-500 mt-1 overflow-hidden">
                            {f.desc}
                          </motion.p>
                        )}
                      </AnimatePresence>
                    </div>
                    {activeFeature === i && (
                      <motion.div layoutId="activeBar" className="w-1 h-8 rounded-full flex-shrink-0"
                        style={{ background: f.accent }} />
                    )}
                  </div>
                </button>
              ))}
            </div>

            <div className="sticky top-24">
              <AnimatePresence mode="wait">
                <motion.div key={activeFeature}
                  initial={{ opacity: 0, x: 24, scale: 0.97 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: -24, scale: 0.97 }}
                  transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                  className="rounded-3xl p-7 min-h-60"
                  style={{ background: features[activeFeature].color }}>
                  <div className="flex items-center gap-2 mb-5">
                    <span className="text-2xl">{features[activeFeature].icon}</span>
                    <h3 className="font-bold text-gray-800">{features[activeFeature].title}</h3>
                  </div>
                  {features[activeFeature].preview}
                </motion.div>
              </AnimatePresence>
              <div className="flex justify-center gap-2 mt-4">
                {features.map((f, i) => (
                  <button key={i} onClick={() => setActiveFeature(i)}
                    className="w-2 h-2 rounded-full transition-all duration-300"
                    style={{
                      background: i === activeFeature ? features[activeFeature].accent : '#d1d5db',
                      transform: i === activeFeature ? 'scale(1.5)' : 'scale(1)',
                    }} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── BENTO GRID ──────────────────────────────────────────────────── */}
      <section className="py-28 px-5">
        <div className="max-w-7xl mx-auto">
          <FadeUp className="text-center mb-14">
            <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-950 tracking-tight">Everything you need.</h2>
            <p className="text-4xl sm:text-5xl font-extrabold mt-1 tracking-tight">
              <span className="text-gray-400">Built to be </span>
              <span className="text-gray-950">fast &amp; smart.</span>
            </p>
          </FadeUp>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {bentoItems.map((item, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ delay: (i % 4) * 0.07, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                whileHover={{ y: -4, scale: 1.02 }}
                className={`bg-gray-50 hover:bg-white border border-gray-200 hover:border-gray-300 hover:shadow-lg rounded-2xl p-5 cursor-default transition-colors group ${item.wide ? 'md:col-span-2' : ''}`}>
                <div className="text-2xl mb-3 group-hover:scale-110 transition-transform inline-block">{item.icon}</div>
                <h3 className="font-bold text-gray-900 text-sm mb-1">{item.title}</h3>
                <p className="text-xs text-gray-500 leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
            <motion.button
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.28, duration: 0.5 }}
              whileHover={{ y: -4, scale: 1.02 }}
              onClick={onEnterApp}
              className="text-white rounded-2xl p-5 transition-all text-left md:col-span-2 group"
              style={{ background: 'linear-gradient(135deg, #d4a017, #b8880f)' }}>
              <div className="text-2xl mb-3 group-hover:scale-110 transition-transform inline-block">✦</div>
              <h3 className="font-bold text-lg mb-1">Explore the app →</h3>
              <p className="text-sm leading-relaxed" style={{ color: 'rgba(255,255,255,0.75)' }}>Try every AI feature — completely free, no sign-up needed</p>
            </motion.button>
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────────── */}
      <section className="py-28 px-5 bg-gray-50">
        <div className="max-w-3xl mx-auto text-center">
          <FadeUp>
            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-950 leading-[1.1] tracking-tight">
              The{' '}
              <span className="relative inline-block">
                <span className="relative z-10 text-[#d4a017]">smarter way</span>
                <motion.span initial={{ scaleX: 0 }} whileInView={{ scaleX: 1 }} viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  className="absolute inset-x-0 bottom-1 h-3 rounded origin-left"
                  style={{ background: 'rgba(212,160,23,0.2)' }} />
              </span>{' '}
              to discover{' '}
              <span style={{ color: '#d4a017' }}>movies</span>
            </h2>
          </FadeUp>
          <FadeUp delay={0.1}>
            <p className="mt-6 text-lg text-gray-500 font-medium max-w-md mx-auto">
              <span className="text-gray-800 font-bold">6,000+ films</span> indexed. Four AI models ready. No account, no credit card.
            </p>
          </FadeUp>
          <FadeUp delay={0.2}>
            <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
              <button onClick={onEnterApp}
                className="px-8 py-4 text-base font-bold text-gray-900 bg-[#d4a017] hover:bg-[#b8880f] rounded-xl transition-all shadow-lg hover:shadow-xl hover:-translate-y-0.5 active:scale-95">
                Start exploring free
              </button>
              <button onClick={onEnterApp}
                className="px-8 py-4 text-base font-bold text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 hover:border-gray-300 rounded-xl transition-all hover:-translate-y-0.5">
                See the Oracle in action
              </button>
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── FOOTER ──────────────────────────────────────────────────────── */}
      <footer style={{ backgroundColor: '#080810' }} className="text-gray-300 px-5 py-16">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-start gap-12 mb-12">
            {/* brand */}
            <div className="md:w-56 flex-shrink-0">
              <div className="flex items-center gap-2 font-extrabold text-xl text-white mb-3">
                <div className="w-7 h-7 rounded-lg bg-[#d4a017] flex items-center justify-center">
                  <svg viewBox="0 0 24 24" fill="white" className="w-4 h-4">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                </div>
                StreamSage
              </div>
              <p className="text-sm text-gray-500 leading-relaxed">AI-powered movie discovery — Oracle RAG, Sentiment BERT, Binge LSTM, Llama 3.</p>
              <div className="flex gap-3 mt-5">
                {[
                  'M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.747l7.73-8.835L1.254 2.25H8.08l4.253 5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z',
                  'M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z',
                  'M23.498 6.186a3.016 3.016 0 00-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 00.502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 002.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 002.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z',
                ].map((d, i) => (
                  <a key={i} href="#"
                    className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                    style={{ backgroundColor: 'rgba(255,255,255,0.08)' }}
                    onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.16)'}
                    onMouseLeave={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.08)'}>
                    <svg viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5 text-gray-300">
                      <path d={d} />
                    </svg>
                  </a>
                ))}
              </div>
            </div>

            {/* links */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-10 flex-1">
              {[
                { heading: 'App',       links: ['Movie Discovery','Oracle RAG','Sentiment Analysis','Binge Predictor','Search filters'] },
                { heading: 'AI Stack',  links: ['Llama 3 LoRA','BERT CUDA','LSTM Model','ChromaDB','BGE Embeddings'] },
                { heading: 'More',      links: ['API Gateway','Health status','Architecture','Source on GitHub','How it works'] },
              ].map(col => (
                <div key={col.heading}>
                  <h4 className="text-xs font-bold uppercase tracking-widest text-gray-600 mb-4">{col.heading}</h4>
                  <ul className="space-y-2.5">
                    {col.links.map(link => (
                      <li key={link}>
                        <a href="#" className="text-sm text-gray-400 hover:text-white transition-colors">{link}</a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          <div className="border-t pt-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-gray-600"
            style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
            <span>© {new Date().getFullYear()} StreamSage. Oracle RAG · Binge LSTM · Sentiment BERT · Llama 3</span>
            <div className="flex gap-5">
              {['Privacy','Terms','Open Source'].map(l => (
                <a key={l} href="#" className="hover:text-gray-400 transition-colors">{l}</a>
              ))}
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
