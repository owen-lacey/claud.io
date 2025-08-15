# FPL Team Picker (Auto FPL)

Auto FPL is a data-driven tool for Fantasy Premier League that helps you optimize your team selection.

## 🎯 Features

- **Optimal Team Calculation**: Suggests the best possible team configuration using mathematical optimization.
- **Transfer Suggestions**: Analyzes your current team and recommends optimal player transfers, considering costs and potential gains.
- **Wildcard Helper**: Builds an entirely new optimized team when you use your wildcard chip.
- **Expected Points (xP) Analysis**: Leverages xP data for informed decision-making.
- **Real-time Data**: Connects to the official FPL API for up-to-date information.

## 🛠️ Technology Stack

- **Backend**: .NET 8, C#, ASP.NET Core, MediatR
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS
- **Data Science**: Python, Jupyter Notebooks, Pandas/NumPy

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
Once running, visit `http://localhost:5079/swagger` for interactive API documentation.

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and entertainment purposes. Fantasy Premier League involves an element of luck, and no algorithm can guarantee success. Always make your own informed decisions!
