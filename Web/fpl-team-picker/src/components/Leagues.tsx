import { useCallback, useContext, useEffect, useState } from "react";
import { DataContext } from "../App";
import { FplApi } from "../helpers/fpl-api";
import { RivalTeam } from "../models/rival-league";
import { LoadingCard } from "./utils/Loading";

function Leagues({ rivalTeams, setRivalTeams }: { rivalTeams: RivalTeam[], setRivalTeams: (teams: RivalTeam[]) => void }) {
  const allData = useContext(DataContext);
  const [leagueId, setLeagueId] = useState<number | null>(null);
  const setLeague = useCallback((leagueToSelect: number) => {
    if (leagueId == leagueToSelect) {
      setLeagueId(null);
    } else {
      setLeagueId(leagueToSelect);
    }
  }, [leagueId]);

  const loadRivalTeams = async (selectedLeague: number) => {
    setRivalTeams([]);
    const league = allData?.leagues.output?.find(l => l.id == selectedLeague);
    const api = new FplApi();
    const results = await Promise.all(league!.participants?.map(async p => {
      const result = await api.users.currentTeamList(p.userId!);
      return new RivalTeam(p, result.data);
    }) || []);
    setRivalTeams(results);
  };

  useEffect(() => {
    if (leagueId != null) {
      loadRivalTeams(leagueId!);
    }
  }, [leagueId]);

  if (!allData?.myDetails.output || !allData?.leagues.output) {
    return <LoadingCard />;
  }
  const { leagues, myDetails } = allData;

  return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4">
    <h2 className="text-xl font-semibold mb-4">My Leagues</h2>
    <div className="flex flex-col">
      <div className="text-gray-500 text-xs"> Select a league &#x1F447;</div>
      {leagues.output!.map((league, i) => {
        const selected = leagueId === league.id;
        let cls = 'relative flex mt-2 p-4 cursor-pointer sexy-container items-end border border-gray-200 hover:bg-gray-200 z-1';
        if (selected) {
          cls += ' bg-gray-200 rounded-t-md';
        } else {
          cls += ' rounded-md'
        }
        return <div key={i}>
          <div onClick={() => setLeague(league.id!)} className={cls}>
            <div className="px-2 flex-1">{league!.name} {selected ? <span>&#x2705;</span> : <></>}</div>
            <div className="px-2 text-xs text-gray-500 font-mono uppercase">Pos: {league.currentPosition}/{league.numberOfPlayers}</div>
          </div>
          <div className={selected ? 'p-2 border border-gray-200 rounded-b-md' : ' opacity-0 h-0'}>

            <table className="w-100">
              <tbody>
                {rivalTeams?.map((rt, i) => {
                  const startingXi = rt.team.startingXi!.map(p => p.player!);
                  const bench = rt.team.bench!.map(p => p.player!);
                  const xp = startingXi!.reduce((a, b) => a + b.xpNext!, 0);
                  const xpBenchBoost = xp + bench!.reduce((a, b) => a + b.xpNext!, 0);
                  const isCurrentUser = rt.rival.userId == myDetails.output!.id;

                  return <tr key={i}>
                    <td>
                      {i <= 2 ? <span>{String.fromCodePoint(129351 + i)}</span> : <></>}
                    </td>
                    <td className="">
                      {isCurrentUser ? <div className="font-semibold">You</div> : <div>{rt.rival.playerName}</div>}
                    </td>
                    <td className="text-gray-500 font-mono text-sm text-right">{Intl.NumberFormat('en-GB').format(rt.rival.total!)}</td>
                    {isCurrentUser ? <></> : <td className="text-blue-500 font-mono text-sm text-blue text-right">{xp.toFixed(1)} ({xpBenchBoost.toFixed(1)})</td>}

                  </tr>
                })}
              </tbody>
            </table>
          </div>
        </div>
      })}
    </div>
  </div>
}

export default Leagues;