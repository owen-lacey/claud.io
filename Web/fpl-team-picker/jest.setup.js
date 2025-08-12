// Jest setup file
require('@testing-library/jest-dom')

// Load environment variables from .env.local for Jest
require('dotenv').config({ path: '.env.local' })

// Mock ResizeObserver for HeadlessUI components
global.ResizeObserver = class ResizeObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock window.location globally if window exists and location doesn't exist or is configurable
if (typeof window !== 'undefined') {
  const locationDescriptor = Object.getOwnPropertyDescriptor(window, 'location');
  if (!locationDescriptor || locationDescriptor.configurable !== false) {
    delete window.location;
    Object.defineProperty(window, 'location', {
      value: {
        reload: jest.fn(),
        href: 'http://localhost:3000',
        origin: 'http://localhost:3000',
        pathname: '/',
        search: '',
        hash: '',
        assign: jest.fn(),
        replace: jest.fn(),
      },
      writable: true,
      configurable: true,
    });
  }
}

// Mock localStorage globally
if (typeof window !== 'undefined') {
  const localStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage');
  if (!localStorageDescriptor || localStorageDescriptor.configurable !== false) {
    delete window.localStorage;
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn(),
      },
      writable: true,
      configurable: true,
    });
  }
}

// Don't override GOOGLE_GENERATIVE_AI_API_KEY if it's already set
// This allows tests to use the real API key from .env.local
if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
  console.log('⚠️ No GOOGLE_GENERATIVE_AI_API_KEY found - API tests will be skipped')
} else {
  console.log('✅ GOOGLE_GENERATIVE_AI_API_KEY loaded for tests')
}