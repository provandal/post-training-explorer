import { useTranslation } from 'react-i18next'

export default function Welcome() {
  const { t } = useTranslation()

  return (
    <div className="max-w-3xl mx-auto text-center">
      <div className="grid grid-cols-2 gap-4 text-left max-w-xl mx-auto">
        <div className="p-3 rounded-lg bg-yellow-950/20 border border-yellow-800/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-yellow-400 mb-1 whitespace-nowrap">
            {t('stop.welcome.rag')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.welcome.ragDesc')}</p>
        </div>
        <div className="p-3 rounded-lg bg-cyan-950/20 border border-cyan-800/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-cyan-400 mb-1 whitespace-nowrap">
            {t('stop.welcome.allTheOptions')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.welcome.allTheOptionsDesc')}</p>
        </div>
        <div className="p-3 rounded-lg bg-orange-950/20 border border-orange-800/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-orange-400 mb-1 whitespace-nowrap">
            {t('stop.welcome.promptEngineering')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.welcome.promptEngineeringDesc')}</p>
        </div>
        <div className="p-3 rounded-lg bg-slate-800 border border-slate-600/30 min-h-[80px]">
          <h4 className="text-sm font-semibold text-slate-200 mb-1 whitespace-nowrap">
            {t('stop.welcome.postTraining')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.welcome.postTrainingDesc')}</p>
        </div>
      </div>
    </div>
  )
}
