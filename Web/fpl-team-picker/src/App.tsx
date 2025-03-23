import './App.scss'
import { useLocalStorage } from "@uidotdev/usehooks";
import AuthGuard from "./components/AuthGuard.tsx";
import Header from "./components/Header.tsx";
import { createContext, memo, useEffect, useState } from "react";
import MyTeam from './components/MyTeam.tsx';
import Players from './components/Players.tsx';
import { FplApi } from './helpers/fpl-api.ts';
import { AllData } from './models/all-data.ts';
import Leagues from './components/Leagues.tsx';
export const DataContext = createContext<AllData | null>(null);
export const SelectedLeagueContext = createContext<number | null>(null);

const App = memo(function App() {
    const [plProfile, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);
    const [data, setData] = useState<AllData | null>(null);
    const [leagueId, setLeagueId] = useState<number | null>(null);

    const loadData = async () => {
        const [myTeam, players, teams, leagues, myDetails] = await Promise.all([
            new FplApi().myTeam.myTeamList(),
            new FplApi().players.playersList(),
            new FplApi().teams.teamsList(),
            new FplApi().myLeagues.myLeaguesList(),
            new FplApi().myDetails.myDetailsList(),
        ]);
        const dataToSet = new AllData(myTeam.data, players.data, teams.data, leagues.data, myDetails.data);
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
            <SelectedLeagueContext.Provider value={leagueId}>
                <div className="app-container">
                    <div className="header">
                        <Header />
                    </div>
                    <div className="my-team">
                        <MyTeam />
                    </div>
                    <div className="leagues">
                        <Leagues leagueId={leagueId} setLeagueId={setLeagueId} />
                    </div>
                    <div className="players">
                        <Players />
                    </div>
                </div>
            </SelectedLeagueContext.Provider>
        </DataContext.Provider>
    )
});

export default App
