import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import HttpBackend from 'i18next-http-backend'
import LanguageDetector from 'i18next-browser-languagedetector'

i18n
  .use(HttpBackend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    supportedLngs: [
      'en',
      'ht',
      'ar-LB',
      'zh-CN',
      'he',
      'es',
      'fr',
      'de',
      'hi',
      'pt-BR',
      'ja',
      'ru',
      'ko',
      'uk',
    ],
    fallbackLng: 'en',
    ns: ['ui'],
    defaultNS: 'ui',

    backend: {
      loadPath: `${import.meta.env.BASE_URL}locales/{{lng}}/{{ns}}.json`,
    },

    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: 'pte_locale',
      caches: ['localStorage'],
    },

    interpolation: {
      escapeValue: false,
    },

    debug: true, // TODO: remove after verifying i18n loading works
  })

export default i18n
