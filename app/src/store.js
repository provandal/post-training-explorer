import { create } from 'zustand'

const useStore = create((set, get) => ({
  // Mode: 'landing' | 'tour' | 'explore'
  mode: 'landing',

  // Tour state
  currentStep: 0,
  tourCompleted: false,

  // Active quadrant for explore mode
  activeQuadrant: null, // 'prompt' | 'rag' | 'posttraining' | 'alloptions'
  activeSubStop: null,  // 'sft' | 'dpo' | 'grpo' | null

  // User interactions tracked during tour
  userPreferences: [],  // For DPO stop - which outputs they preferred
  userPrompts: [],      // Custom prompts they've tried

  // Model loading state (for future live inference)
  modelsLoaded: {
    base: false,
    sft: false,
    dpo: false,
    grpo: false,
  },

  // Actions
  startTour: () => set({ mode: 'tour', currentStep: 0 }),
  startExplore: () => set({ mode: 'explore', activeQuadrant: 'prompt' }),

  nextStep: () => {
    const { currentStep } = get()
    const maxStep = 15 // Total tour steps (0-indexed)
    if (currentStep < maxStep) {
      set({ currentStep: currentStep + 1 })
    } else {
      set({ tourCompleted: true, mode: 'explore' })
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

  setModelLoaded: (model) =>
    set((state) => ({
      modelsLoaded: { ...state.modelsLoaded, [model]: true },
    })),
}))

export default useStore
