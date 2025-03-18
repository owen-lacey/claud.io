import './App.scss'
import { useLocalStorage } from "@uidotdev/usehooks";
import AuthGuard from "./components/AuthGuard.tsx";
import Header from "./components/Header.tsx";
import { createContext, memo, useEffect, useState } from "react";
import MyTeam from './components/MyTeam.tsx';
import Players from './components/Players.tsx';
import { FplApi } from './helpers/fpl-api.ts';
import { AllData } from './models/all-data.ts';
export const DataContext = createContext<AllData | null>(null);

const App = memo(function App() {
    const [plProfile, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);
    const [data, setData] = useState<AllData | null>(null);

    const loadData = async () => {
        const myTeam = await new FplApi().myTeam.myTeamList();
        const players = await new FplApi().players.playersList();
        const teams = await new FplApi().teams.teamsList();
        const dataToSet = new AllData(myTeam.data, players.data, teams.data);
        setData(dataToSet);
    }

    useEffect(() => {
        loadData();
    }, []);

    if (!plProfile) {
        return <AuthGuard onDone={(cookie) => savePlProfile(cookie)} />
    }

    return (
        <DataContext.Provider value={data}>
            <div className="app-container">
                <div className="header">
                    <Header />
                </div>
                <div className="my-team">
                    <MyTeam />
                </div>
                <div className="players">
                    <Players />
                </div>
                <div className="leagues"></div>
            </div>
        </DataContext.Provider>
    )
});

export default App
