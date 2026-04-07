import { useTranslation, Trans } from 'react-i18next'
import useStore from '../store'
import LanguageSelector from './LanguageSelector'

export default function Landing() {
  const { t } = useTranslation()
  const startTour = useStore((s) => s.startTour)
  const startExplore = useStore((s) => s.startExplore)
  const startTrain = useStore((s) => s.startTrain)
  const startResults = useStore((s) => s.startResults)

  const cards = [
    {
      key: 'guidedTour',
      title: t('landing.guidedTour'),
      description: t('landing.guidedTourDesc'),
      action: startTour,
      buttonLabel: t('landing.startTour'),
      buttonClass:
        'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-600/30 hover:shadow-blue-500/40',
    },
    {
      key: 'exploreFree',
      title: t('landing.exploreFree'),
      description: t('landing.exploreFreeDesc'),
      action: startExplore,
      buttonLabel: t('landing.explore'),
      buttonClass: 'bg-slate-600 hover:bg-slate-500 shadow-lg shadow-slate-600/20',
    },
    {
      key: 'resultsLearnings',
      title: t('landing.resultsLearnings'),
      description: t('landing.resultsLearningsDesc'),
      action: startResults,
      buttonLabel: t('landing.seeResults'),
      buttonClass:
        'bg-purple-600 hover:bg-purple-500 shadow-lg shadow-purple-600/30 hover:shadow-purple-500/40',
    },
    {
      key: 'trainYourModel',
      title: t('landing.trainYourModel'),
      description: t('landing.trainYourModelDesc'),
      action: startTrain,
      buttonLabel: t('landing.getStarted'),
      buttonClass:
        'bg-emerald-600 hover:bg-emerald-500 shadow-lg shadow-emerald-600/30 hover:shadow-emerald-500/40',
    },
  ]

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10">
      {/* Language selector */}
      <div className="absolute top-4 right-4">
        <LanguageSelector />
      </div>

      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-5xl font-extrabold text-white mb-3 tracking-tight">
          {t('landing.title')}
        </h1>
        <p className="text-lg text-slate-400 max-w-xl mx-auto leading-relaxed">
          <Trans i18nKey="landing.subtitle">
            An interactive guide to the techniques that turn a general-purpose language model into
            one that works for
            <span className="text-cyan-400 font-semibold">your</span> organization.
          </Trans>
        </p>
      </div>

      {/* Three cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl w-full mb-10">
        {cards.map((card) => (
          <div
            key={card.key}
            className="bg-slate-800/50 border border-slate-700/50 hover:border-blue-600/50 rounded-xl p-6 flex flex-col transition-colors"
          >
            <h2 className="text-xl font-bold text-white mb-3">{card.title}</h2>
            <p className="text-sm text-slate-400 leading-relaxed flex-1 mb-5">{card.description}</p>
            <button
              onClick={card.action}
              className={`w-full py-3 text-white rounded-lg font-semibold text-sm transition-all hover:scale-[1.02] active:scale-[0.98] ${card.buttonClass}`}
            >
              {card.buttonLabel}
            </button>
          </div>
        ))}
      </div>

      {/* Bottom details */}
      <div className="flex items-center gap-6 text-xs text-slate-600">
        <span>{t('landing.footerTask')}</span>
        <span className="w-1 h-1 rounded-full bg-slate-700" />
        <span>{t('landing.footerModel')}</span>
        <span className="w-1 h-1 rounded-full bg-slate-700" />
        <span>{t('landing.footerReal')}</span>
      </div>
      <p className="mt-4 text-[10px] text-slate-700">
        Optimization framework adapted from{' '}
        <a
          href="https://www.youtube.com/watch?v=ahnGLM-RC1Y"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-slate-500"
        >
          &ldquo;A Survey of Techniques for Maximizing LLM Performance&rdquo;
        </a>{' '}
        by Colin Jarvis &amp; John Allard, OpenAI DevDay 2023.
      </p>
    </div>
  )
}
