import { useCallback, useContext } from "react";
import { DataContext } from "../App";
import "../styles/sexy.scss";

function Leagues({leagueId, setLeagueId}: {leagueId: number | null, setLeagueId: (leagueToSelect: number | null) => void}) {
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
    const { leagues } = allData;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">My Leagues</h2>
            <div className="flex flex-col">
                <div className="text-gray-500 text-xs"> Select a league &#x1F447;</div>
                {leagues.map((league, i) => {
                    const selected = leagueId === league.id;
                    let cls = 'flex rounded-md my-2 p-4 cursor-pointer sexy-container items-end border border-gray-200 hover:bg-gray-200';
                    if (selected) {
                        cls += ' bg-gray-200';
                    }
                    return <div onClick={_ => setLeague(league.id!)} key={i} className={cls}>
                        <div className="px-2 flex-1">{league!.name} {selected ? <span>&#x2705;</span> : <></>}</div>
                        <div className="px-2 text-xs text-gray-500 font-mono uppercase">Pos: {league.currentPosition}/{league.numberOfPlayers}</div>
                    </div>
                })}
            </div>
        </div>
}

export default Leagues;