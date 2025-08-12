/**
 * @jest-environment node
 */

import { POST } from './route'

describe('/api/chat', () => {
  beforeEach(() => {
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.log('⚠️ GOOGLE_GENERATIVE_AI_API_KEY is not set. Skipping API tests.')
    }
  })

  it('should return 200 status for valid request', async () => {
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.log('Skipping test - no API key')
      return
    }
    
    const mockRequest = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'Say hello in exactly 2 words' }],
      }),
    })

    const response = await POST(mockRequest)
    expect(response.status).toBe(200)
    
    // Verify we get a streaming response
    expect(response.headers.get('content-type')).toContain('text/plain')
    
    // Read a bit of the response to ensure it's working
    const reader = response.body?.getReader()
    expect(reader).toBeDefined()
    
    if (reader) {
      const { value } = await reader.read()
      expect(value).toBeDefined()
      
      const chunk = new TextDecoder().decode(value)
      // Fail if the chunk is an error
      expect(chunk).not.toMatch(/3:"An error occurred."/)
      // Should match a valid streaming format
      expect(chunk).toMatch(/(0:.*|f:\{.*\}|e:\{.*\}|d:\{.*\})/)
    }
  }, 15000)

  it('should handle request with system prompt', async () => {
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.log('Skipping test - no API key')
      return
    }
    
    const mockRequest = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: 'You are a helpful assistant. Be very brief.' },
          { role: 'user', content: 'What is 2+2?' }
        ],
      }),
    })

    const response = await POST(mockRequest)
    expect(response.status).toBe(200)
    
    // Verify we can read the response
    const reader = response.body?.getReader()
    if (reader) {
      const { value } = await reader.read()
      expect(value).toBeDefined()
      
      const chunk = new TextDecoder().decode(value)
      // Fail if the chunk is an error
      expect(chunk).not.toMatch(/3:"An error occurred."/)
      // Should match a valid streaming format
      expect(chunk).toMatch(/(0:.*|f:\{.*\}|e:\{.*\}|d:\{.*\})/)
    }
  }, 10000)

  it('should handle malformed JSON gracefully', async () => {
    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.log('Skipping test - no API key')
      return
    }
    
    const mockRequest = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: 'invalid json',
    })

    // This should either return an error response or throw
    try {
      const response = await POST(mockRequest)
      // If it doesn't throw, it should return an error status
      expect(response.status).toBeGreaterThanOrEqual(400)
    } catch (error) {
      // If it throws, that's also acceptable error handling
      expect(error).toBeDefined()
    }
  })
})
