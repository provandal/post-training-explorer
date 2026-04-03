import { Component, useState, useRef, useEffect, lazy, Suspense } from 'react'
import { useTranslation } from 'react-i18next'

const GeoLingua = lazy(() =>
  import('geolingua').catch(() => ({
    default: () => null,
  })),
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

function GeoLinguaFallback() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/90">
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 text-center max-w-sm">
        <p className="text-slate-400 text-sm">
          GeoLingua is not available. Use the language dropdown instead.
        </p>
      </div>
    </div>
  )
}

function GeoLinguaErrorBoundary({ children, fallback }) {
  const [hasError, setHasError] = useState(false)

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setHasError(false), [children])

  if (hasError) return fallback

  return <ErrorCatcher onError={() => setHasError(true)}>{children}</ErrorCatcher>
}

class ErrorCatcher extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch() {
    this.props.onError?.()
  }

  render() {
    if (this.state.hasError) return null
    return this.props.children
  }
}

export default function LanguageSelector() {
  const { i18n } = useTranslation()
  const [open, setOpen] = useState(false)
  const [globeOpen, setGlobeOpen] = useState(false)
  const wrapperRef = useRef(null)

  const currentLang = LANGUAGES.find((l) => l.code === i18n.language) || LANGUAGES[0]

  /* Close dropdown when clicking outside */
  useEffect(() => {
    function handleClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  function selectLanguage(code) {
    i18n.changeLanguage(code)
    localStorage.setItem('pte_locale', code)
    setOpen(false)
  }

  function handleGeoLinguaSelect(code) {
    if (LANGUAGES.some((l) => l.code === code)) {
      selectLanguage(code)
    }
    setGlobeOpen(false)
  }

  return (
    <>
      <div ref={wrapperRef} className="relative inline-block">
        {/* Language button */}
        <button
          onClick={() => setOpen((prev) => !prev)}
          className="bg-slate-800 border border-slate-700 text-slate-400 hover:border-blue-500 hover:text-slate-200 rounded-md px-2.5 py-1.5 text-xs transition-colors"
          aria-label="Select language"
        >
          {currentLang.flag} {currentLang.label}
        </button>

        {/* Dropdown */}
        {open && (
          <div className="absolute right-0 mt-1 w-56 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 py-1 max-h-80 overflow-y-auto">
            {LANGUAGES.map((lang) => {
              const isActive = lang.code === i18n.language
              return (
                <button
                  key={lang.code}
                  onClick={() => selectLanguage(lang.code)}
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

            {/* GeoLingua globe trigger inside dropdown */}
            <div className="border-t border-slate-700 mt-1 pt-1 px-3 py-2">
              <button
                onClick={() => {
                  setOpen(false)
                  setGlobeOpen(true)
                }}
                className="w-8 h-8 rounded-full border-2 border-slate-700 bg-slate-800 hover:border-blue-500 flex items-center justify-center transition-colors mx-auto"
                aria-label="Open GeoLingua globe"
              >
                🌐
              </button>
            </div>
          </div>
        )}
      </div>

      {/* GeoLingua full-screen modal overlay */}
      {globeOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/95">
          <div className="relative w-full h-full flex items-center justify-center">
            {/* Close button */}
            <button
              onClick={() => setGlobeOpen(false)}
              className="absolute top-4 right-4 z-10 bg-slate-800 border border-slate-700 text-slate-400 hover:border-blue-500 hover:text-slate-200 rounded-md px-3 py-1.5 text-sm transition-colors"
            >
              ✕ Close
            </button>

            <GeoLinguaErrorBoundary fallback={<GeoLinguaFallback />}>
              <Suspense
                fallback={<div className="text-slate-400 text-sm">Loading GeoLingua...</div>}
              >
                <GeoLingua
                  onSelect={handleGeoLinguaSelect}
                  supportedLanguages={LANGUAGES.map((l) => l.code)}
                />
              </Suspense>
            </GeoLinguaErrorBoundary>
          </div>
        </div>
      )}
    </>
  )
}
