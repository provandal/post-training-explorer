import { describe, it, expect } from 'vitest'
import { isLoaded, getCategories, formatPromptMetrics, getTestPrompts } from './loadArtifacts'

describe('loadArtifacts', () => {
  it('isLoaded returns false before any fetch', () => {
    expect(isLoaded()).toBe(false)
  })

  it('getCategories returns 6 fallback categories when not loaded', () => {
    const categories = getCategories()
    expect(categories).toHaveLength(6)
    expect(categories).toContain('OLTP Database')
    expect(categories).toContain('Backup Archive')
  })

  it('getTestPrompts returns empty array when not loaded', () => {
    expect(getTestPrompts()).toEqual([])
  })

  it('formatPromptMetrics handles null gracefully', () => {
    expect(formatPromptMetrics(null)).toBe('')
    expect(formatPromptMetrics(undefined)).toBe('')
  })

  it('formatPromptMetrics returns prompt text when no metrics', () => {
    expect(formatPromptMetrics({ prompt: 'test prompt' })).toBe('test prompt')
  })
})
