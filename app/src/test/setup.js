import '@testing-library/jest-dom'
import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

i18n.use(initReactI18next).init({
  lng: 'en',
  resources: {
    en: {
      ui: {
        'landing.title': 'Post-Training Explorer',
      },
    },
  },
  ns: ['ui'],
  defaultNS: 'ui',
  interpolation: { escapeValue: false },
})
