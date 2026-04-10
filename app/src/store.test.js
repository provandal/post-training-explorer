import { describe, it, expect, beforeEach } from 'vitest'
import useStore from './store'

describe('useStore', () => {
  beforeEach(() => {
    useStore.setState({
      mode: 'landing',
      currentStep: 0,
      tourCompleted: false,
      tourReturnStep: null,
      activeQuadrant: null,
      activeSubStop: null,
      userPreferences: [],
      selectedPromptId: 0,
    })
  })

  it('has correct initial state', () => {
    const state = useStore.getState()
    expect(state.mode).toBe('landing')
    expect(state.currentStep).toBe(0)
    expect(state.tourCompleted).toBe(false)
    expect(state.selectedPromptId).toBe(0)
    expect(state.userPreferences).toEqual([])
  })

  it('startTour sets mode to tour and step to 0', () => {
    useStore.getState().startTour()
    const state = useStore.getState()
    expect(state.mode).toBe('tour')
    expect(state.currentStep).toBe(0)
  })

  it('startExplore sets mode to explore', () => {
    useStore.getState().startExplore()
    expect(useStore.getState().mode).toBe('explore')
    expect(useStore.getState().activeQuadrant).toBe('prompt')
  })

  it('startTrain sets mode to train', () => {
    useStore.getState().startTrain()
    expect(useStore.getState().mode).toBe('train')
  })

  it('startResults sets mode to results', () => {
    useStore.getState().startResults()
    expect(useStore.getState().mode).toBe('results')
  })

  it('nextStep increments currentStep', () => {
    useStore.getState().startTour()
    useStore.getState().nextStep()
    expect(useStore.getState().currentStep).toBe(1)
  })

  it('nextStep does not exceed max step (11) and transitions to explore', () => {
    useStore.setState({ mode: 'tour', currentStep: 11 })
    useStore.getState().nextStep()
    // At max step, nextStep transitions to explore mode
    expect(useStore.getState().mode).toBe('explore')
    expect(useStore.getState().tourCompleted).toBe(true)
  })

  it('prevStep decrements currentStep', () => {
    useStore.setState({ currentStep: 5 })
    useStore.getState().prevStep()
    expect(useStore.getState().currentStep).toBe(4)
  })

  it('prevStep does not go below 0', () => {
    useStore.setState({ currentStep: 0 })
    useStore.getState().prevStep()
    expect(useStore.getState().currentStep).toBe(0)
  })

  it('goToStep sets currentStep directly', () => {
    useStore.getState().goToStep(10)
    expect(useStore.getState().currentStep).toBe(10)
  })

  it('setSelectedPromptId updates selected prompt', () => {
    useStore.getState().setSelectedPromptId(7)
    expect(useStore.getState().selectedPromptId).toBe(7)
  })

  it('addPreference appends to userPreferences', () => {
    useStore.getState().addPreference({ chosen: 'A', rejected: 'B' })
    useStore.getState().addPreference({ chosen: 'C', rejected: 'D' })
    expect(useStore.getState().userPreferences).toHaveLength(2)
  })

  it('setActiveQuadrant sets quadrant and optional subStop', () => {
    useStore.getState().setActiveQuadrant('posttraining', 'sft')
    const state = useStore.getState()
    expect(state.activeQuadrant).toBe('posttraining')
    expect(state.activeSubStop).toBe('sft')
  })
})
