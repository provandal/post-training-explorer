import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Treat geolingua as optional — if not installed, resolve to an empty module
// so the build succeeds and the app falls back to the dropdown language selector.
function optionalDep(name) {
  const virtualId = `\0optional:${name}`
  let isInstalled = false
  try {
    // eslint-disable-next-line no-undef
    isInstalled = !!require.resolve(name)
  } catch {
    /* not installed */
  }
  return {
    name: `optional-dep-${name}`,
    resolveId(id) {
      if (id === name && !isInstalled) return virtualId
    },
    load(id) {
      if (id === virtualId) return 'export default null;'
    },
  }
}

export default defineConfig({
  plugins: [optionalDep('geolingua'), react(), tailwindcss()],
  base: './',
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
  },
})
