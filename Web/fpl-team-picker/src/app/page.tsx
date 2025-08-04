"use client";

import '../App.scss'
// import { useLocalStorage } from "@uidotdev/usehooks";
import { useState } from "react";
import AuthGuard from "../components/AuthGuard";
import Header from "../components/Header";
import { memo, useEffect } from "react";
import MyTeam from '../components/MyTeam';
import Players from '../components/Players';
import { FplApi } from '../helpers/fpl-api';
import { AllData } from '../models/all-data';
import Leagues from '../components/Leagues';
import { RivalTeam } from '../models/rival-league';
import SmallScreen from '../components/utils/SmallScreen';
import { ApiResult } from '../models/api-result';
import { MyTeam as ApiMyTeam, League, Player, Team, User } from '../helpers/api';
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from '@assistant-ui/react-ai-sdk';
import { DataContext, RivalTeamsContext } from '../lib/contexts';

const App = memo(function App() {
  // Temporarily comment out useLocalStorage to isolate the issue
  // const [plProfile, savePlProfile] = useLocalStorage<string | null>("pl_profile", null);
  const [plProfile, setPlProfile] = useState<string | null>(null);
  const savePlProfile = (value: string | null) => {
    setPlProfile(value);
    if (typeof window !== 'undefined') {
      if (value) {
        localStorage.setItem("pl_profile", value);
      } else {
        localStorage.removeItem("pl_profile");
      }
    }
  };

  const [data, setData] = useState<AllData | null>(null);
  const [rivalTeams, setRivalTeams] = useState<RivalTeam[]>([]);
  
  const runtime = useChatRuntime({
    api: "/api/chat",
  });

  const loadData = async () => {
    const [myTeam, players, teams, leagues, myDetails] = await Promise.all([
      new FplApi().myTeam.myTeamList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<ApiMyTeam>(false, err)),
      new FplApi().players.playersList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<Player[]>(false, err)),
      new FplApi().teams.teamsList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<Team[]>(false, err)),
      new FplApi().myLeagues.myLeaguesList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<League[]>(false, err)),
      new FplApi().myDetails.myDetailsList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<User>(false, err)),
    ]);
    const dataToSet = new AllData(myTeam, players, teams, leagues, myDetails);
    setData(dataToSet);
  }

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem("pl_profile");
      if (stored) {
        setPlProfile(stored);
      }
    }
  }, []);

  useEffect(() => {
    if (plProfile != null) {
      loadData();
    }
  }, [plProfile]);

  let content;
  if (!plProfile) {
    content = <AuthGuard onDone={(cookie) => savePlProfile(cookie)} />;
  } else {
    content = (
      <AssistantRuntimeProvider runtime={runtime}>
        <div className={`app-container`}>
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
        </div>
      </AssistantRuntimeProvider>
    );
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
