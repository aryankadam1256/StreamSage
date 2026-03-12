import { useState, useEffect, useRef } from 'react'
import { motion, useInView, AnimatePresence } from 'framer-motion'

const FadeUp = ({ children, delay = 0, className = '' }) => {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 32 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

const logos = [
  { name: 'Census',    style: 'font-bold tracking-tight' },
  { name: 'Epsor',     style: 'font-semibold italic' },
  { name: 'figures',   style: 'font-light tracking-widest uppercase text-sm' },
  { name: 'pennylane', style: 'font-bold' },
  { name: 'airfocus',  style: 'font-semibold tracking-tight' },
  { name: 'GetAccept', style: 'font-bold tracking-tight' },
  { name: 'HubSpot',   style: 'font-black tracking-tight' },
  { name: 'Livestorm', style: 'font-bold' },
]

const features = [
  {
    icon: '💬',
    title: 'More engaging Q&A sessions',
    desc: 'Let attendees upvote questions and keep the conversation flowing naturally with live audience participation.',
    color: '#dcfce7',
    accent: '#16a34a',
    preview: (
      <div className="space-y-3">
        {['What is the pricing model?', 'Can I export recordings?', 'Is there a free trial?'].map((q, i) => (
          <div key={i} className="flex items-center gap-3 bg-white rounded-xl p-3 shadow-sm border border-gray-100">
            <div className="w-7 h-7 rounded-full bg-green-100 flex items-center justify-center text-green-700 font-bold text-xs flex-shrink-0">{i + 1}</div>
            <span className="text-sm text-gray-700 flex-1">{q}</span>
            <span className="text-xs text-green-600 font-bold">{12 - i * 3} ▲</span>
          </div>
        ))}
      </div>
    ),
  },
  {
    icon: '👁️',
    title: 'Keep people watching',
    desc: 'Smart pacing tools and interactive moments prevent drop-off and maintain peak attention.',
    color: '#fef9c3',
    accent: '#ca8a04',
    preview: (
      <div className="space-y-3">
        <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
          <p className="text-xs text-gray-400 mb-2 font-medium">Audience Retention</p>
          <div className="flex items-end gap-1 h-14">
            {[90,85,88,82,91,87,84,89,93,88,92,95].map((v,i) => (
              <div key={i} className="flex-1 rounded-t bg-amber-400" style={{ height: `${v}%`, opacity: 0.8 }} />
            ))}
          </div>
        </div>
        <div className="bg-amber-50 rounded-xl p-3 border border-amber-100 text-xs text-amber-800 font-semibold">
          ✨ 93% avg retention — 2× industry average
        </div>
      </div>
    ),
  },
  {
    icon: '🎨',
    title: 'Branded webinars in 2 clicks',
    desc: 'Match your brand perfectly with custom colors, logos, and themes applied instantly.',
    color: '#ede9fe',
    accent: '#7c3aed',
    preview: (
      <div className="space-y-3">
        <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
          <p className="text-xs text-gray-400 mb-3 font-medium">Brand Kit</p>
          <div className="flex gap-2 mb-3">
            {['#7c3aed','#16a34a','#0ea5e9','#f59e0b','#ef4444'].map((c) => (
              <div key={c} className="w-7 h-7 rounded-lg border-2 border-white shadow cursor-pointer" style={{ background: c }} />
            ))}
          </div>
          <div className="h-8 rounded-lg flex items-center px-3" style={{ background: 'linear-gradient(90deg,#7c3aed,#6d28d9)' }}>
            <span className="text-white text-xs font-bold">Your Company Webinar</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: '✨',
    title: 'Chat has never been this fun',
    desc: 'Reactions, GIFs, and threaded discussions keep energy high throughout.',
    color: '#fce7f3',
    accent: '#db2777',
    preview: (
      <div className="space-y-2">
        {[
          { name: 'Sarah', msg: 'This is amazing! 🎉' },
          { name: 'James', msg: '100% agree 👏' },
          { name: 'Priya', msg: '🔥🔥🔥' },
          { name: 'Marco', msg: 'Can you share the slides?' },
        ].map((m, i) => (
          <div key={i} className="flex items-start gap-2">
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-pink-400 to-purple-500 text-white text-xs flex items-center justify-center font-bold flex-shrink-0">
              {m.name[0]}
            </div>
            <div className="bg-white rounded-xl px-3 py-1.5 shadow-sm border border-gray-100">
              <span className="text-xs font-semibold text-gray-700">{m.name} </span>
              <span className="text-xs text-gray-600">{m.msg}</span>
            </div>
          </div>
        ))}
      </div>
    ),
  },
  {
    icon: '🌟',
    title: 'Webinars that feel alive',
    desc: 'Dynamic transitions and real-time audience data make every session memorable.',
    color: '#e0f2fe',
    accent: '#0284c7',
    preview: (
      <div className="grid grid-cols-2 gap-2">
        {['Live Polls','Screen Share','Breakouts','Recording'].map((t) => (
          <div key={t} className="bg-white rounded-xl p-3 shadow-sm border border-gray-100 text-center">
            <div className="text-blue-500 text-lg mb-1">⚡</div>
            <p className="text-xs font-semibold text-gray-700">{t}</p>
          </div>
        ))}
      </div>
    ),
  },
]

const bentoItems = [
  { icon: '📧', title: 'Branded emails',      desc: 'Auto-send styled confirmations & reminders', wide: true },
  { icon: '🎨', title: 'Brand kit',           desc: 'One-click theming across all assets' },
  { icon: '🙋', title: 'Q&A',                 desc: 'Upvote-driven live questions' },
  { icon: '⏰', title: 'Automated reminders', desc: 'Perfect timing, every time' },
  { icon: '📊', title: 'Polls',               desc: 'Real-time audience insight' },
  { icon: '📈', title: 'Viewer engagement',   desc: 'Heatmaps & drop-off analytics' },
  { icon: '📋', title: 'Registration pages',  desc: 'Beautiful, conversion-optimized sign-up pages', wide: true },
  { icon: '🎬', title: 'On-demand webinars',  desc: 'Evergreen content that works while you sleep' },
  { icon: '🔗', title: 'CRM integrations',    desc: 'HubSpot, Salesforce, Zapier & more' },
  { icon: '📡', title: '1080p streaming',     desc: 'Crystal-clear video for every attendee' },
  { icon: '📺', title: 'Webinar channel',     desc: 'Your own branded content hub' },
  { icon: '📉', title: 'Webinar analytics',   desc: 'Deep data on every session', wide: true },
]

const avatarGradients = [
  'from-green-400 to-emerald-600',
  'from-blue-400 to-cyan-600',
  'from-purple-400 to-pink-600',
  'from-amber-400 to-orange-500',
  'from-rose-400 to-red-600',
]

const ProductMockup = () => (
  <div className="w-full rounded-2xl overflow-hidden shadow-[0_32px_80px_-12px_rgba(0,0,0,0.18)] border border-gray-200">
    <div className="h-9 bg-gray-100 border-b border-gray-200 flex items-center px-4 gap-1.5 flex-shrink-0">
      <div className="w-3 h-3 rounded-full bg-red-400" />
      <div className="w-3 h-3 rounded-full bg-amber-400" />
      <div className="w-3 h-3 rounded-full bg-green-400" />
      <div className="flex-1 mx-4 h-5 bg-white rounded border border-gray-200 flex items-center justify-center">
        <span className="text-xs text-gray-400">app.contrast.video/live</span>
      </div>
    </div>
    <div className="bg-gray-950 p-4 flex gap-3" style={{ minHeight: 280 }}>
      <div className="flex-1 rounded-xl bg-gradient-to-br from-gray-800 to-gray-900 relative overflow-hidden flex flex-col justify-end p-3">
        <div className="absolute inset-0 flex items-center justify-center opacity-20 text-6xl">🎙️</div>
        <div className="absolute top-3 left-3 flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
          <span className="text-white text-xs font-medium">LIVE</span>
        </div>
        <motion.div
          className="w-full h-1 bg-gray-700 rounded-full overflow-hidden"
        >
          <motion.div
            className="h-full bg-green-400 rounded-full"
            animate={{ width: ['30%','70%','45%','80%','35%','60%'] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          />
        </motion.div>
        <div className="absolute bottom-6 left-3 bg-black/60 rounded-lg px-2 py-0.5">
          <span className="text-white text-xs font-medium">Alex Presenter</span>
        </div>
      </div>
      <div className="w-52 flex flex-col gap-2">
        <div className="grid grid-cols-2 gap-1">
          {['Priya K.','James L.','Anna M.','Sam T.'].map((name, i) => (
            <div key={i} className="rounded-lg bg-gray-800 flex flex-col items-end justify-end p-1.5 relative" style={{ minHeight: 64 }}>
              <div className="absolute inset-0 flex items-center justify-center text-xl opacity-40">
                {['🧑','👩','👧','🧔'][i]}
              </div>
              <div className="bg-black/60 rounded px-1 py-0.5 w-full">
                <span className="text-white text-xs block truncate">{name}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="bg-gray-900 rounded-xl p-2 flex-1 overflow-hidden">
          <p className="text-xs text-gray-400 font-medium mb-1.5">Chat · 124 watching</p>
          {['Love this! 🔥','Super helpful 👏','Question!'].map((msg, i) => (
            <div key={i} className="flex items-start gap-1 mb-1">
              <div className={`w-4 h-4 rounded-full flex-shrink-0 bg-gradient-to-br ${avatarGradients[i]}`} />
              <p className="text-xs text-gray-300 leading-tight">{msg}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
    <div className="bg-gray-900 border-t border-gray-800 px-4 py-2.5 flex items-center justify-between">
      <div className="flex gap-2">
        {['🎤','📷','📺','🙋'].map((e, i) => (
          <button key={i} className="w-8 h-8 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors flex items-center justify-center text-sm">
            {e}
          </button>
        ))}
      </div>
      <button className="px-4 py-1.5 rounded-lg bg-red-600 text-white text-xs font-semibold">End session</button>
    </div>
  </div>
)

const AIMockup = () => (
  <div className="rounded-2xl overflow-hidden shadow-[0_24px_60px_-8px_rgba(0,0,0,0.13)] border border-gray-200 bg-white">
    <div className="h-8 bg-gray-50 border-b border-gray-200 flex items-center px-4 gap-1.5">
      <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
      <div className="w-2.5 h-2.5 rounded-full bg-amber-400" />
      <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
    </div>
    <div className="p-5 space-y-3">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-6 h-6 rounded-md bg-green-500 flex items-center justify-center text-white text-xs">✨</div>
        <span className="text-sm font-semibold text-gray-700">AI Content Suite</span>
        <span className="ml-auto text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">4 assets ready</span>
      </div>
      {[
        { icon: '📄', label: 'Summary blog post',  sub: '1,200 words · SEO-optimized' },
        { icon: '✂️', label: 'Short video clips',   sub: '8 clips · auto-captioned' },
        { icon: '📨', label: 'Newsletter edition',  sub: '5 sections · ready to send' },
        { icon: '🐦', label: 'Social media kit',    sub: '12 posts · all platforms' },
      ].map((item, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -12 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.1, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          className="flex items-center gap-3 p-3 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          <span className="text-lg">{item.icon}</span>
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-800">{item.label}</p>
            <p className="text-xs text-gray-400">{item.sub}</p>
          </div>
          <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">✓</div>
        </motion.div>
      ))}
    </div>
  </div>
)

export default function LandingPage({ onEnterApp }) {
  const [scrolled,       setScrolled]       = useState(false)
  const [activeFeature,  setActiveFeature]  = useState(0)
  const [mobileOpen,     setMobileOpen]     = useState(false)

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

      {/* ── NAVBAR ─────────────────────────────────────────────────────── */}
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-white/90 backdrop-blur-md shadow-sm border-b border-gray-100' : 'bg-white/80 backdrop-blur-sm'
      }`}>
        <div className="max-w-7xl mx-auto px-5 h-16 flex items-center gap-4">
          <a href="#" className="flex items-center gap-2 font-bold text-xl tracking-tight text-gray-900 mr-2">
            <div className="w-7 h-7 rounded-lg bg-green-500 flex items-center justify-center">
              <svg viewBox="0 0 24 24" fill="white" className="w-4 h-4">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
            </div>
            contrast
          </a>

          <nav className="hidden md:flex items-center gap-1 flex-1">
            {['Features','Pricing','Videos','Learn','Free Tools'].map(item => (
              <a key={item} href="#"
                className="px-3.5 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-50 transition-all">
                {item}
              </a>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <a href="#" className="hidden md:flex px-4 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-50 rounded-lg transition-all">
              Book a demo
            </a>
            <button onClick={onEnterApp}
              className="px-4 py-2 text-sm font-semibold text-white bg-green-600 hover:bg-green-700 rounded-lg transition-all shadow-sm hover:shadow-md active:scale-95">
              Start for free
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
              {['Features','Pricing','Videos','Learn','Free Tools','Book a demo'].map(item => (
                <a key={item} href="#" className="block py-2.5 text-sm font-medium text-gray-700 border-b border-gray-50 hover:text-gray-900">{item}</a>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* ── HERO ───────────────────────────────────────────────────────── */}
      <section className="pt-32 pb-0 px-5 text-center overflow-hidden">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-green-50 border border-green-200 text-sm font-semibold text-green-700 mb-8">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          New: AI Content Repurposing is here
          <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd"/>
          </svg>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          {['Effortless webinars,', 'powerfully engaging'].map((line, li) => (
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
          Engaging features that make your audience smile.{' '}
          <span className="text-gray-700">Infinite content</span> with repurpose AI magic. Easy as ever.
        </motion.p>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
          <button onClick={onEnterApp}
            className="px-7 py-3.5 text-base font-bold text-white bg-green-600 hover:bg-green-700 rounded-xl transition-all shadow-lg hover:shadow-xl hover:-translate-y-0.5 active:scale-95">
            Start for free
          </button>
          <button className="px-7 py-3.5 text-base font-bold text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 hover:border-gray-300 rounded-xl transition-all hover:-translate-y-0.5">
            Book a demo
          </button>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.65, duration: 0.6 }}
          className="mt-5 flex items-center justify-center gap-3">
          <div className="flex -space-x-2">
            {avatarGradients.map((cls, i) => (
              <div key={i} className={`w-8 h-8 rounded-full border-2 border-white bg-gradient-to-br ${cls}`} />
            ))}
          </div>
          <p className="text-sm text-gray-500 font-medium">
            Loved by <span className="text-gray-900 font-bold">10,000+</span> teams
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

      {/* ── LOGO TICKER ────────────────────────────────────────────────── */}
      <section className="py-14 border-t border-b border-gray-100 mt-16 overflow-hidden bg-gray-50/60">
        <p className="text-center text-xs font-semibold uppercase tracking-widest text-gray-400 mb-8">
          Trusted by the world's best teams
        </p>
        <div className="relative flex overflow-hidden">
          <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-gray-50 to-transparent z-10 pointer-events-none" />
          <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-gray-50 to-transparent z-10 pointer-events-none" />
          <div className="logo-ticker flex gap-16 items-center whitespace-nowrap">
            {[...logos, ...logos].map((logo, i) => (
              <span key={i} className={`text-xl text-gray-400 hover:text-gray-700 transition-colors cursor-default select-none ${logo.style}`}>
                {logo.name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── AI CONTENT ─────────────────────────────────────────────────── */}
      <section className="py-28 px-5 overflow-hidden">
        <div className="relative overflow-hidden mb-14 py-2.5 bg-green-50 border-y border-green-100">
          <div className="ai-ticker flex gap-8 whitespace-nowrap text-green-600 text-xs font-bold uppercase tracking-widest">
            {Array.from({ length: 24 }).map((_, i) => (
              <span key={i} className="flex items-center gap-2 select-none">
                <span className="w-1 h-1 rounded-full bg-green-400 inline-block" />
                new
              </span>
            ))}
          </div>
        </div>

        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <FadeUp>
              <span className="inline-block px-3 py-1 rounded-full bg-green-50 border border-green-200 text-xs font-bold uppercase tracking-widest text-green-700 mb-5">
                AI Repurposing
              </span>
              <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-950 leading-[1.1] tracking-tight">
                Get a week&apos;s worth of content out of every webinar with{' '}
                <span className="relative inline-block">
                  <span className="relative z-10">Ai</span>
                  <span className="absolute inset-x-0 bottom-1 h-3 bg-green-200 opacity-60 rounded" />
                </span>
              </h2>
              <p className="mt-5 text-lg text-gray-500 leading-relaxed">
                Turn every session into a summary blog, short clips, a polished newsletter, and a full social kit — automatically.
              </p>
            </FadeUp>
            <FadeUp delay={0.15}>
              <div className="mt-7 flex flex-wrap gap-2">
                {['📄 Blog post','✂️ Video clips','📨 Newsletter','🐦 Social kit','📱 Short-form'].map(tag => (
                  <span key={tag} className="px-3 py-1.5 rounded-full bg-gray-100 text-sm font-semibold text-gray-700 hover:bg-green-50 hover:text-green-800 transition-colors cursor-default">
                    {tag}
                  </span>
                ))}
              </div>
            </FadeUp>
            <FadeUp delay={0.25}>
              <button onClick={onEnterApp}
                className="mt-8 px-6 py-3 text-sm font-bold text-white bg-green-600 hover:bg-green-700 rounded-xl transition-all shadow hover:shadow-lg hover:-translate-y-0.5 active:scale-95">
                Try it free →
              </button>
            </FadeUp>
          </div>
          <FadeUp delay={0.1}><AIMockup /></FadeUp>
        </div>
      </section>

      {/* ── KEEP YOUR AUDIENCE ─────────────────────────────────────────── */}
      <section className="py-28 px-5 bg-gray-50/60">
        <div className="max-w-7xl mx-auto">
          <FadeUp className="text-center mb-16">
            <span className="text-3xl mb-3 block">🎯</span>
            <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-950 tracking-tight">
              Keep your audience until the end
            </h2>
            <p className="mt-4 text-lg text-gray-500 max-w-lg mx-auto">
              Every feature designed to hold attention and delight attendees from intro to outro.
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

      {/* ── BENTO GRID ─────────────────────────────────────────────────── */}
      <section className="py-28 px-5">
        <div className="max-w-7xl mx-auto">
          <FadeUp className="text-center mb-14">
            <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-950 tracking-tight">Everything you need.</h2>
            <p className="text-4xl sm:text-5xl font-extrabold mt-1 tracking-tight">
              <span className="text-gray-400">Designed to be </span>
              <span className="text-gray-950">fast &amp; easy.</span>
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
              className="bg-green-600 hover:bg-green-700 text-white rounded-2xl p-5 transition-all text-left md:col-span-2 group">
              <div className="text-2xl mb-3 group-hover:scale-110 transition-transform inline-block">🚀</div>
              <h3 className="font-bold text-lg mb-1">All features →</h3>
              <p className="text-sm text-green-100 leading-relaxed">Explore every tool included on the free plan</p>
            </motion.button>
          </div>
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────────────────────────── */}
      <section className="py-28 px-5 bg-gray-50">
        <div className="max-w-3xl mx-auto text-center">
          <FadeUp>
            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-950 leading-[1.1] tracking-tight">
              The{' '}
              <span className="relative inline-block">
                <span className="relative z-10 text-green-600">new way</span>
                <motion.span initial={{ scaleX: 0 }} whileInView={{ scaleX: 1 }} viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  className="absolute inset-x-0 bottom-1 h-3 bg-green-200 rounded origin-left" />
              </span>{' '}
              to run{' '}
              <span className="text-green-600">webinars</span>
            </h2>
          </FadeUp>
          <FadeUp delay={0.1}>
            <p className="mt-6 text-lg text-gray-500 font-medium max-w-md mx-auto">
              Go live with <span className="text-gray-800 font-bold">50 registrants</span> on the free plan.{' '}
              No credit card needed.
            </p>
          </FadeUp>
          <FadeUp delay={0.2}>
            <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
              <button onClick={onEnterApp}
                className="px-8 py-4 text-base font-bold text-white bg-green-600 hover:bg-green-700 rounded-xl transition-all shadow-lg hover:shadow-xl hover:-translate-y-0.5 active:scale-95">
                Start for free
              </button>
              <button className="px-8 py-4 text-base font-bold text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 hover:border-gray-300 rounded-xl transition-all hover:-translate-y-0.5">
                Book a demo
              </button>
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── FOOTER ─────────────────────────────────────────────────────── */}
      <footer style={{ backgroundColor: '#0a1a09' }} className="text-gray-300 px-5 py-16">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-start gap-12 mb-12">
            <div className="md:w-52 flex-shrink-0">
              <div className="flex items-center gap-2 font-bold text-xl text-white mb-3">
                <div className="w-7 h-7 rounded-lg bg-green-500 flex items-center justify-center">
                  <svg viewBox="0 0 24 24" fill="white" className="w-4 h-4">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                  </svg>
                </div>
                contrast
              </div>
              <p className="text-sm text-gray-500 leading-relaxed">Fun, engaging and authentic webinars.</p>
              <div className="flex gap-3 mt-5">
                {[
                  'M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.747l7.73-8.835L1.254 2.25H8.08l4.253 5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z',
                  'M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z',
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

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-10 flex-1">
              {[
                { heading: 'Product',   links: ['Features','HubSpot Integration',"What's new",'Pricing','Log in'] },
                { heading: 'Resources', links: ['Learn','Videos','Help Center','Webinar Glossary'] },
                { heading: 'More',      links: ['Our story',"We're hiring",'Contrast vs. Livestorm','Contrast vs. Zoom'] },
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
            <span>© {new Date().getFullYear()} Contrast. All rights reserved.</span>
            <div className="flex gap-5">
              {['Privacy Policy','Terms of Service','Cookies'].map(l => (
                <a key={l} href="#" className="hover:text-gray-400 transition-colors">{l}</a>
              ))}
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
