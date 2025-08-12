"use client";

import { useCallback, useContext, useEffect, useState } from "react";
import { FplApi } from "../../helpers/fpl-api";
import { RivalTeam } from "../../models/rival-league";
import { LoadingCard } from "../utils/Loading";
import { DataContext } from "@/lib/contexts";
import { LeagueParticipant, SelectedSquad } from "../../helpers/api";

interface LeaguesProps {
  rivalTeams: RivalTeam[];
  setRivalTeams: (teams: RivalTeam[]) => void;
  plProfile?: string;
}

function Leagues({ rivalTeams, setRivalTeams, plProfile }: LeaguesProps) {
  const allData = useContext(DataContext);
  const [selectedLeagueIdx, setSelectedLeagueIdx] = useState(-1);

  const getOutput = useCallback((field: any) => {
    if (field.loading) {
      return [];
    }

    if (field.error) {
      return [];
    }

    return field.output || [];
  }, []);

  useEffect(() => {
    if (selectedLeagueIdx === -1 || !allData?.leagues.output || !allData?.myDetails.output) {
      setRivalTeams([]);
      return;
    }

    const selectedLeague = allData.leagues.output[selectedLeagueIdx];
    if (!selectedLeague?.participants) {
      setRivalTeams([]);
      return;
    }

    // Fetch team data for each participant
    const fetchRivalTeams = async () => {
      try {
        const api = new FplApi(plProfile || undefined);
        const teamPromises = selectedLeague.participants!.map(async (participant: LeagueParticipant) => {
          try {
            const teamResponse = await api.users.currentTeamList(participant.userId!);
            return new RivalTeam(participant, teamResponse.data);
          } catch (error) {
            console.error(`Failed to fetch team for user ${participant.userId}:`, error);
            return null;
          }
        });

        const teams = await Promise.all(teamPromises);
        const validTeams = teams.filter((team): team is RivalTeam => team !== null);
        setRivalTeams(validTeams);
      } catch (error) {
        console.error('Failed to fetch rival teams:', error);
        setRivalTeams([]);
      }
    };

    fetchRivalTeams();
  }, [selectedLeagueIdx, allData?.leagues.output, allData?.myDetails.output, setRivalTeams, plProfile]);

  if (!allData?.myDetails.output || !allData?.leagues.output) {
    return <LoadingCard />;
  }

  const { leagues, myDetails } = allData;

  return (
    <div className="bg-card border border-border shadow-lg rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-2 text-foreground">Leagues</h2>
      {getOutput(leagues).map((league: any, idx: number) => {
        let selected = selectedLeagueIdx === idx;
        let cls = 'relative flex mt-2 p-4 cursor-pointer sexy-container items-end border border-border hover:bg-accent hover:text-accent-foreground z-1';
        if (selected) {
          cls += ' bg-accent text-accent-foreground rounded-t-md';
        }
        return (
          <div key={league.id}>
            <div
              className={cls}
              onClick={() => { setSelectedLeagueIdx(selected ? -1 : idx) }}>
              <strong className="flex-grow">{league.name}</strong>
            </div>
            <div className={selected ? 'p-2 border border-border bg-card rounded-b-md' : ' opacity-0 h-0'}>
              <table className="w-full">
                <tbody>
                  {rivalTeams?.map((rt, i) => {
                    const startingXi = rt.team.startingXi || [];
                    const bench = rt.team.bench || [];
                    const xp = startingXi.reduce((a, b) => a + (b.player?.xp || 0), 0);
                    const xpBenchBoost = xp + bench.reduce((a, b) => a + (b.player?.xp || 0), 0);
                    const isCurrentUser = rt.rival.userId == myDetails.output!.id;

                    return (
                      <tr key={i}>
                        <td>
                          {i <= 2 ? <span>{String.fromCodePoint(129351 + i)}</span> : <></>}
                        </td>
                        <td className="text-left px-2">
                          <div className={isCurrentUser ? 'font-bold text-foreground' : 'text-foreground'}>
                            {isCurrentUser ? 'You' : rt.rival.playerName}
                          </div>
                        </td>
                        <td className="text-muted-foreground font-mono text-sm text-right">
                          {Intl.NumberFormat('en-GB').format(rt.rival.total || 0)}
                        </td>
                        {!isCurrentUser && (
                          <td className="text-primary font-mono text-sm text-right">
                            {xp.toFixed(1)} ({xpBenchBoost.toFixed(1)})
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default Leagues;