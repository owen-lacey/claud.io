/**
 * @jest-environment node
 */

// Mock data service so tools mode doesn't hit network if the model decides to call a tool
jest.mock('@/lib/data-service', () => ({
  dataService: {
    async getSelectionContext() {
      return {
        players: [],
        teams: [],
        myTeam: { freeTransfers: 1, bank: 0, budget: 1000, squad: null },
        user: { id: 1, name: 'Test' },
      } as any;
    },
  },
}));

import { POST } from './route'

describe('/api/chat (tools mode)', () => {
  beforeEach(() => {
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.log('⚠️ GOOGLE_GENERATIVE_AI_API_KEY is not set. Skipping API tests.')
    }
  })

  it('should stream in tools mode (text or error event), returning 200', async () => {
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.log('Skipping test - no API key')
      return
    }
    
    const mockRequest = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'tools',
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Brief hello.' }
        ],
      }),
    })

    const response = await POST(mockRequest)
    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toContain('text/plain')

    const reader = response.body?.getReader()
    expect(reader).toBeDefined()

    if (reader) {
      const { value } = await reader.read()
      expect(value).toBeDefined()
      const chunk = new TextDecoder().decode(value)
      // Accept either normal streaming tokens or an error event token.
      // This verifies the streaming contract without flaking on transient API issues.
      expect(chunk).toMatch(/(0:.*|f:\{.*\}|e:\{.*\}|d:\{.*\}|3:.*)/)
    }
  }, 15000)
})
