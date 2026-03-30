import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('renders landing page by default', () => {
    render(<App />)
    expect(screen.getByText('Post-Training Explorer')).toBeInTheDocument()
  })
})
