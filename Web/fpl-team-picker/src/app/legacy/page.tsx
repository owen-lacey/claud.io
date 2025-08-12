"use client";

import '../../components/legacy/App.scss'
// import { useLocalStorage } from "@uidotdev/usehooks";
import { memo, useCallback, useEffect, useState } from "react";
import AuthGuard from "../../components/AuthGuard";
import Header from "../../components/Header";
import MyTeam from '../../components/legacy/MyTeam';
import Players from '../../components/legacy/Players';
import Chat from '../../components/legacy/Chat';
import { FplApi } from '../../helpers/fpl-api';
import { AllData } from '../../models/all-data';
import Leagues from '../../components/legacy/Leagues';
import { RivalTeam } from '../../models/rival-league';
import SmallScreen from '../../components/utils/SmallScreen';
import { ApiResult } from '../../models/api-result';
import { MyTeam as ApiMyTeam, League, Player, Team, User } from '../../helpers/api';
import { DataContext, RivalTeamsContext } from '../../lib/contexts';

const LegacyApp = memo(function LegacyApp() {
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

  const loadData = useCallback(async () => {
    const api = new FplApi(plProfile || undefined);
    const [myTeam, players, teams, leagues, myDetails] = await Promise.all([
      api.myTeam.myTeamList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<ApiMyTeam>(false, err)),
      api.players.playersList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<Player[]>(false, err)),
      api.teams.teamsList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<Team[]>(false, err)),
      api.myLeagues.myLeaguesList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<League[]>(false, err)),
      api.myDetails.myDetailsList().then(res => new ApiResult(true, res.data)).catch(err => new ApiResult<User>(false, err)),
    ]);
    const dataToSet = new AllData(myTeam, players, teams, leagues, myDetails);
    setData(dataToSet);
  }, [plProfile]);

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
  }, [plProfile, loadData]);

  let content;
  if (!plProfile) {
    content = <AuthGuard onDone={(cookie) => savePlProfile(cookie)} />;
  } else {
    content = (
        <div className={`app-container`}>
          <div className="header">
            <Header />
          </div>
          <div className="chat">
            <Chat />
          </div>
          <div className="my-team">
            <MyTeam />
          </div>
          <div className="leagues">
            <Leagues rivalTeams={rivalTeams} setRivalTeams={setRivalTeams} plProfile={plProfile || undefined} />
          </div>
          <div className="players">
            <Players />
          </div>
        </div>
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

export default LegacyApp
