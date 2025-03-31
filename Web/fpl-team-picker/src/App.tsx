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
import { RivalTeam } from './models/rival-league.ts';
import SmallScreen from './components/utils/SmallScreen.tsx';
export const DataContext = createContext<AllData | null>(null);
export const RivalTeamsContext = createContext<RivalTeam[]>([]);

const App = memo(function App() {
  const [plProfile, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);
  const [data, setData] = useState<AllData | null>(null);
  const [rivalTeams, setRivalTeams] = useState<RivalTeam[]>([]);

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
    if (plProfile != null) {
      loadData();
    }
  }, [plProfile]);

  let content;
  if (!plProfile) {
    content = <AuthGuard onDone={(cookie) => savePlProfile(cookie)} />;
  } else {
    content = <div className="app-container">
      <div className="header">
        <Header />
      </div>
      <div className="my-team">
        <MyTeam />
      </div>
      <div className="leagues">
        <Leagues rivalTeams={rivalTeams} setRivalTeams={setRivalTeams} />
      </div>
      <div className="players">
        <Players />
      </div>
    </div>;
  }

  return (
    <DataContext.Provider value={data}>
      <RivalTeamsContext.Provider value={rivalTeams}>
        <div className="md:hidden">
          <SmallScreen />
        </div>
        <div className="hidden md:block">
          {content}
        </div>
      </RivalTeamsContext.Provider>
    </DataContext.Provider>
  )
});

export default App
