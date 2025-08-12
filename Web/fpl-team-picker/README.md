# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

# 🏆 FPL Team Picker

AI-powered Fantasy Premier League assistant with smart squad building and transfer suggestions.

## ✨ Features

- **🤖 AI Chat Assistant**: Get personalized FPL advice and squad recommendations
- **⚽ Smart Squad Builder**: Build optimal squads within budget constraints
- **🔄 Transfer Suggestions**: AI-powered transfer recommendations
- **💾 Squad Management**: Save, load, and manage multiple squad configurations
- **📊 Player Analytics**: Expected points, form analysis, and team insights
- **🎯 Interactive Tools**: Captain selection, bench optimization, and formation tweaks

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Start development server**
   ```bash
   npm run dev
   ```

3. **Open your browser**
   Navigate to `http://localhost:3000`

## 🏗️ Tech Stack

- **Frontend**: React 18 + TypeScript + Next.js 14
- **Styling**: Tailwind CSS + Custom SCSS
- **AI Integration**: OpenAI API with streaming responses
- **State Management**: React Context + Local Storage
- **Testing**: Jest + React Testing Library + Playwright
- **Build**: Next.js with optimized production builds

## 🧪 Testing

```bash
# Run unit tests
npm test

# Run component tests
npm run test:components

# Run E2E tests
npm run test:e2e

# Run all tests
npm run test:all
```

## 📁 Project Structure

```
src/
├── app/                 # Next.js app router pages
├── components/          # React components
│   ├── assistant-ui/    # Chat interface components
│   ├── team/           # Squad management components
│   └── ui/             # Reusable UI components
├── helpers/            # API clients and utilities
├── lib/               # Core services and contexts
├── models/            # TypeScript type definitions
└── styles/            # Global styles and themes
```

## 🎮 Usage

1. **Navigate to `/assistant`** for the main chat interface
2. **Build a squad** using the "Build Squad" quick action
3. **Chat with the AI** for personalized FPL advice
4. **Save your squads** for future reference
5. **Export/import** your data for backup

## 🔧 Development

- **Hot reload**: Changes reflect instantly during development
- **Type safety**: Full TypeScript coverage with strict mode
- **Code quality**: ESLint + Prettier for consistent formatting
- **Testing**: Comprehensive test coverage across the stack

## 📊 API Integration

The app integrates with FPL API endpoints for real-time data:
- Player statistics and form
- Team information and fixtures
- User team data and transfers
- League standings and rankings

---

**Built with ❤️ for FPL managers who want that extra edge!** ⚽️

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config({
  plugins: {
    // Add the react-x and react-dom plugins
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    // other rules...
    // Enable its recommended typescript rules
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
})
```
