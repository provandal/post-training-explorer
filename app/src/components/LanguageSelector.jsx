import React, { useState, useRef, useEffect, lazy, Suspense } from 'react'
import { useTranslation } from 'react-i18next'

// --- GeoLingua dynamic import (optional dependency) ---
let geoLinguaModule = null
const geoLinguaReady = import('geolingua')
  .then((m) => {
    if (m.GeoLingua) geoLinguaModule = m
  })
  .catch(() => {})

const GeoLinguaLazy = lazy(() =>
  import('geolingua').then((m) => {
    if (!m.GeoLingua) throw new Error('geolingua not installed')
    return { default: m.GeoLingua }
  }),
)

const LANGUAGES = [
  { code: 'en', flag: '🇺🇸', label: 'English' },
  { code: 'es', flag: '🇪🇸', label: 'Español' },
  { code: 'fr', flag: '🇫🇷', label: 'Français' },
  { code: 'de', flag: '🇩🇪', label: 'Deutsch' },
  { code: 'pt-BR', flag: '🇧🇷', label: 'Português (BR)' },
  { code: 'ru', flag: '🇷🇺', label: 'Русский' },
  { code: 'uk', flag: '🇺🇦', label: 'Українська' },
  { code: 'zh-CN', flag: '🇨🇳', label: '中文 (简体)' },
  { code: 'ja', flag: '🇯🇵', label: '日本語' },
  { code: 'ko', flag: '🇰🇷', label: '한국어' },
  { code: 'hi', flag: '🇮🇳', label: 'हिन्दी' },
  { code: 'he', flag: '🇮🇱', label: 'עברית' },
  { code: 'ar-LB', flag: '🇱🇧', label: 'العربية (لبنان)' },
  { code: 'ht', flag: '🇭🇹', label: 'Kreyòl Ayisyen' },
]

// --- Error boundary for GeoLingua ---
class GeoLinguaErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }
  static getDerivedStateFromError() {
    return { hasError: true }
  }
  componentDidCatch() {
    // GeoLingua failed to render — silently fall back
  }
  render() {
    return this.state.hasError ? null : this.props.children
  }
}

// --- GeoLingua globe icon ---
function GeoLinguaIcon({ onLanguageSelect }) {
  const [available, setAvailable] = useState(false)
  const [showGlobe, setShowGlobe] = useState(false)
  const overlayRef = useRef(null)
  const openedAtRef = useRef(0)

  useEffect(() => {
    geoLinguaReady.then(() => {
      if (geoLinguaModule) setAvailable(true)
    })
  }, [])

  if (!available) return null

  const handleSelect = (locale) => {
    onLanguageSelect(locale)
    if (showGlobe && Date.now() - openedAtRef.current > 1000) {
      setTimeout(() => setShowGlobe(false), 600)
    }
  }

  const handleOpen = () => {
    openedAtRef.current = Date.now()
    setShowGlobe(true)
  }

  return (
    <>
      <button
        onClick={handleOpen}
        className="w-7 h-7 rounded-full border border-slate-700 bg-slate-800 hover:border-blue-500 hover:scale-110 flex items-center justify-center transition-all text-sm flex-shrink-0"
        aria-label="Open GeoLingua globe"
        title="GeoLingua"
      >
        🌐
      </button>

      {showGlobe && (
        <div
          ref={overlayRef}
          onClick={(e) => {
            if (e.target === overlayRef.current) setShowGlobe(false)
          }}
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)' }}
        >
          <div
            className="relative rounded-2xl overflow-hidden shadow-2xl"
            style={{ width: 520, maxWidth: '95vw', height: 600, maxHeight: '90vh' }}
          >
            <button
              onClick={() => setShowGlobe(false)}
              className="absolute top-2 right-2 z-10 w-8 h-8 rounded-full border border-slate-700 bg-slate-900/80 text-slate-400 hover:text-white flex items-center justify-center text-lg"
            >
              ×
            </button>
            <GeoLinguaErrorBoundary>
              <Suspense
                fallback={
                  <div className="w-full h-full bg-slate-950 flex items-center justify-center text-slate-500 text-sm">
                    Loading globe...
                  </div>
                }
              >
                <GeoLinguaLazy
                  initialMode="full"
                  theme="space"
                  onLanguageSelect={handleSelect}
                  showSkip={false}
                  voiceDetectionEnabled={true}
                  detectBrowserLanguage={true}
                  persist={false}
                  style={{ width: '100%', height: '100%' }}
                />
              </Suspense>
            </GeoLinguaErrorBoundary>
          </div>
        </div>
      )}
    </>
  )
}

// --- Dropdown selector ---
function DropdownSelector({ onLanguageSelect }) {
  const { i18n } = useTranslation()
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  const current = LANGUAGES.find((l) => l.code === i18n.language) || LANGUAGES[0]

  useEffect(() => {
    if (!open) return
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  const handleSelect = (code) => {
    onLanguageSelect(code)
    setOpen(false)
  }

  return (
    <div ref={ref} className="relative inline-block">
      <button
        onClick={() => setOpen((o) => !o)}
        className="bg-slate-800 border border-slate-700 text-slate-400 hover:border-blue-500 hover:text-slate-200 rounded-md px-2.5 py-1.5 text-xs transition-colors flex items-center gap-1"
        aria-label="Select language"
      >
        <span>{current.flag}</span>
        <span className="hidden sm:inline">{current.label}</span>
        <span className="text-[8px] ms-0.5">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="absolute right-0 mt-1 w-56 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 py-1 max-h-80 overflow-y-auto">
          {LANGUAGES.map((lang) => {
            const isActive = lang.code === i18n.language
            return (
              <button
                key={lang.code}
                onClick={() => handleSelect(lang.code)}
                className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? 'text-blue-400 font-semibold bg-slate-800'
                    : 'text-slate-200 hover:bg-slate-800'
                }`}
              >
                {lang.flag} {lang.label}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// --- Main export ---
export default function LanguageSelector() {
  const { i18n } = useTranslation()

  const handleLanguageSelect = (code) => {
    i18n.changeLanguage(code)
    localStorage.setItem('pte_locale', code)
  }

  return (
    <div className="flex items-center gap-1.5">
      <GeoLinguaIcon onLanguageSelect={handleLanguageSelect} />
      <DropdownSelector onLanguageSelect={handleLanguageSelect} />
    </div>
  )
}
