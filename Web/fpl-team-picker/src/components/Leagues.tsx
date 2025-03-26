import { useCallback, useContext } from "react";
import { DataContext } from "../App";
import "../styles/sexy.scss";
import { lookupPlayer } from "../helpers/lookups";

function Leagues({ leagueId, setLeagueId }: { leagueId: number | null, setLeagueId: (leagueToSelect: number | null) => void }) {
    const allData = useContext(DataContext);
    const setLeague = useCallback((leagueToSelect: number) => {
        if (leagueId == leagueToSelect) {
            setLeagueId(null);
        } else {
            setLeagueId(leagueToSelect);
        }
    }, [leagueId]);

    if (!allData) {
        return <></>;
    }
    const { leagues, players, myDetails } = allData;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4">
        <h2 className="text-xl font-semibold mb-4">My Leagues</h2>
        <div className="flex flex-col">
            <div className="text-gray-500 text-xs"> Select a league &#x1F447;</div>
            {leagues.map((league, i) => {
                const selected = leagueId === league.id;
                let cls = 'relative flex mt-2 p-4 cursor-pointer sexy-container items-end border border-gray-200 hover:bg-gray-200 z-1';
                if (selected) {
                    cls += ' bg-gray-200 rounded-t-md';
                } else {
                    cls += ' rounded-md'
                }
                return <div key={i}>
                    <div onClick={_ => setLeague(league.id!)} className={cls}>
                        <div className="px-2 flex-1">{league!.name} {selected ? <span>&#x2705;</span> : <></>}</div>
                        <div className="px-2 text-xs text-gray-500 font-mono uppercase">Pos: {league.currentPosition}/{league.numberOfPlayers}</div>
                    </div>
                    <div className={selected ? 'p-2 border border-gray-200 rounded-b-md' : ' opacity-0 h-0'}>

                        <table className="w-100">
                            <tbody>
                                {league.participants?.map((p, i) => {
                                    const startingXi = p.startingXi?.map(p => lookupPlayer(p, players));
                                    const bench = p.bench?.map(p => lookupPlayer(p, players));
                                    const xp = startingXi!.reduce((a, b) => a + b.xpNext!, 0);
                                    const xpBenchBoost = xp + bench!.reduce((a, b) => a + b.xpNext!, 0);
                                    const isCurrentUser = p.userId == myDetails.id;

                                    return <tr key={i}>
                                        <td>
                                            {i <= 2 ? <span>{String.fromCodePoint(129351 + i)}</span> : <></>}
                                        </td>
                                        <td className="flex justify-between items-center px-2">
                                            {isCurrentUser ? <div className="font-semibold">You</div> : <div>{p.playerName}</div>}
                                        </td>
                                        <td className="text-gray-500 font-mono text-sm text-right">{Intl.NumberFormat('en-GB').format(p.total!)}</td>
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