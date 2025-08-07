// Jest setup file
require('@testing-library/jest-dom')

// Load environment variables from .env.local for Jest
require('dotenv').config({ path: '.env.local' })

// Don't override GOOGLE_GENERATIVE_AI_API_KEY if it's already set
// This allows tests to use the real API key from .env.local
if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
  console.log('⚠️ No GOOGLE_GENERATIVE_AI_API_KEY found - API tests will be skipped')
} else {
  console.log('✅ GOOGLE_GENERATIVE_AI_API_KEY loaded for tests')
}