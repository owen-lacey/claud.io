# FPL Team Picker (Auto FPL)

Auto FPL is an application that acts as a readonly wrapper around your FPL team. It provides data-driven optimization for Fantasy Premier League team selection using advanced algorithms and statistical analysis.

## 🎯 Main Features

1. **Optimal Team Calculation** - Uses mathematical optimization to suggest the best possible team configuration
2. **Expected Points Analysis** - Leverages xP (expected points) data for informed decision making
3. **Real-time Data Integration** - Connects directly to the official FPL API for up-to-date information

## 🚀 Use Cases

### 1. Calculate Transfers
- Analyzes your current team and suggests optimal player transfers
- Considers transfer costs (-4 points) vs potential point gains
- Evaluates both short-term and long-term transfer strategies
- Takes into account upcoming fixtures, player form, and injury status

### 2. Calculate Wildcard
- Builds an entirely new optimized team when using your wildcard chip
- No transfer cost constraints - complete team reconstruction
- Optimizes budget allocation across all positions (GK, DEF, MID, FWD)
- Considers team diversity and fixture difficulty

### 3. TOTS (Team of the Season)
- Identifies the highest-performing players across the entire season
- Useful for end-of-season analysis and planning
- Helps identify consistent performers vs. differential picks

## 🏗️ Architecture

The application follows a clean architecture pattern with the following components:

### Backend (.NET 8 API)
- **FplTeamPicker.Api** - REST API endpoints and controllers
- **FplTeamPicker.Domain** - Core business entities and contracts
- **FplTeamPicker.Services** - Business logic and use cases
- **FplTeamPicker.Optimisation** - Mathematical optimization algorithms
- **FplTeamPicker.Tests** - Unit and integration tests

### Frontend (React + TypeScript)
- **React 19** with TypeScript for type safety
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Headless UI** for accessible components

### Data Analysis (Python)
- **Jupyter Notebooks** for data exploration and model training
- **Expected Points (xP) Analysis** for player performance prediction
- **Goalkeeper-specific models** for position-based optimization

## 🎲 FPL Rules & Constraints

The optimization algorithm respects all official FPL rules:

### Team Composition
- **15 total players**: 11 starting, 4 bench
- **2 Goalkeepers**: 1 starting, 1 bench
- **5 Defenders**: 3-5 can start
- **5 Midfielders**: 2-5 can start  
- **3 Forwards**: 1-3 can start
- **£100.0m total budget**

### Team Restrictions
- **Maximum 3 players** from any single Premier League team
- **11 starting players** must be selected each gameweek
- **Captain** scores double points, **Vice-captain** as backup

### Transfer System
- **1 free transfer** per gameweek (can bank up to 2)
- **-4 point penalty** for each additional transfer
- **Wildcard chip** allows unlimited free transfers (limited uses)

## 🧠 Optimization Model

### Objective Function
Maximize expected points while respecting all FPL constraints:

```
Maximize: Σ(player_xp * selection_weight) - transfer_penalties
```

### Key Metrics
- **xP (Expected Points)** - Predicted points based on underlying statistics
- **xP per £** - Value efficiency metric
- **Fixture Difficulty Rating** - Upcoming match difficulty
- **Ownership %** - Template vs. differential considerations
- **Form** - Recent performance trends

### Position-Specific Analysis
- **Goalkeepers** - Clean sheet probability, save points, bonus potential
- **Defenders** - Clean sheets, attacking returns, bonus points
- **Midfielders** - Goals, assists, clean sheet points (if playing defense)
- **Forwards** - Goals, assists, penalty taking

## 🛠️ Technology Stack

### Backend
- **.NET 8** - Modern C# framework
- **MediatR** - CQRS pattern implementation
- **ASP.NET Core** - Web API framework
- **HTTP Client** - FPL API integration

### Frontend  
- **React 19** - Latest React with concurrent features
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client for API calls

### Data Science
- **Python** - Data analysis and modeling
- **Jupyter Notebooks** - Interactive data exploration
- **Pandas/NumPy** - Data manipulation and analysis

## 🚦 Getting Started

### Prerequisites
- **.NET 8 SDK**
- **Node.js 18+**
- **Python 3.8+** (for data analysis)

### Backend Setup
```bash
cd Api
dotnet restore
dotnet run --project FplTeamPicker.Api
```

### Frontend Setup
```bash
cd Web/fpl-team-picker
npm install
npm run dev
```

### API Documentation
Once running, visit `http://localhost:5079/swagger` for interactive API documentation.

## 📊 Data Sources

- **Official FPL API** - Player data, fixtures, team information
- **Historical Performance** - Season-long statistics and trends
- **Expected Points Models** - Custom xP calculations based on underlying stats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and entertainment purposes. Fantasy Premier League involves an element of luck, and no algorithm can guarantee success. Always make your own informed decisions!

