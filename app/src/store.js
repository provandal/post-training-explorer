import { create } from 'zustand'
import { loadArtifacts, isLoaded } from './data/loadArtifacts'

const useStore = create((set, get) => ({
  // Mode: 'landing' | 'tour' | 'explore' | 'train' | 'results'
  mode: 'landing',

  // Tour state
  currentStep: 0,
  tourCompleted: false,
  tourReturnStep: null, // Step to return to when coming back from explore deep dives

  // Active quadrant for explore mode
  activeQuadrant: null, // 'prompt' | 'rag' | 'posttraining' | 'alloptions'
  activeSubStop: null, // 'sft' | 'dpo' | 'grpo' | null

  // User interactions tracked during tour
  userPreferences: [], // For DPO stop - which outputs they preferred
  userPrompts: [], // Custom prompts they've tried

  // Precomputed data loading state
  artifactsLoaded: false,
  artifactsError: null,

  // Selected test prompt for pattern picker (0-19)
  selectedPromptId: 0,

  // Model loading state (for client-side inference)
  modelsLoaded: {
    base: false,
    grpo: false,
  },
  modelDownloadProgress: {
    base: 0,
    grpo: 0,
  },

  // Actions
  startTour: () => {
    set({ mode: 'tour', currentStep: 0 })
    get().loadArtifactsIfNeeded()
  },
  startExplore: () => {
    set({ mode: 'explore', activeQuadrant: 'prompt', tourReturnStep: null })
    get().loadArtifactsIfNeeded()
  },
  startTrain: () => {
    set({ mode: 'train' })
    get().loadArtifactsIfNeeded()
  },
  startResults: () => {
    set({ mode: 'results' })
    get().loadArtifactsIfNeeded()
  },

  nextStep: () => {
    const { currentStep } = get()
    const maxStep = 11 // Total tour steps (0-indexed) — 12 steps in tourSteps.js
    if (currentStep < maxStep) {
      set({ currentStep: currentStep + 1 })
    } else {
      set({
        tourCompleted: true,
        mode: 'explore',
        activeQuadrant: 'prompt',
        tourReturnStep: currentStep,
      })
    }
  },

  prevStep: () => {
    const { currentStep } = get()
    if (currentStep > 0) {
      set({ currentStep: currentStep - 1 })
    }
  },

  goToStep: (step) => set({ currentStep: step }),

  setActiveQuadrant: (quadrant, subStop = null) =>
    set({ activeQuadrant: quadrant, activeSubStop: subStop }),

  addPreference: (preference) =>
    set((state) => ({ userPreferences: [...state.userPreferences, preference] })),

  setMode: (mode) => set({ mode }),

  setSelectedPromptId: (id) => set({ selectedPromptId: id }),

  loadArtifactsIfNeeded: async () => {
    if (isLoaded()) {
      set({ artifactsLoaded: true })
      return
    }
    const success = await loadArtifacts()
    set({
      artifactsLoaded: success,
      artifactsError: success ? null : 'Failed to load precomputed data',
    })
  },

  setArtifactsLoaded: (loaded = true) => set({ artifactsLoaded: loaded }),

  setModelLoaded: (model) =>
    set((state) => ({
      modelsLoaded: { ...state.modelsLoaded, [model]: true },
    })),

  setModelDownloadProgress: (model, progress) =>
    set((state) => ({
      modelDownloadProgress: { ...state.modelDownloadProgress, [model]: progress },
    })),
}))

export default useStore
