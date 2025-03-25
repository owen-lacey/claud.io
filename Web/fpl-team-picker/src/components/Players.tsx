import { useContext, useEffect, useState } from "react";
import { DataContext, SelectedLeagueContext } from "../App";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";
import SelectionIcon from "./players/SelectionIcon";
import { PlayerSelection } from "../models/player-selection";
import RivalSelectionCount from "./players/RivalSelectionCount";

function Players() {
    const allData = useContext(DataContext);
    const selectedLeague = useContext(SelectedLeagueContext);
    const [playerSelections, setPlayerSelections] = useState<{ [id: number]: PlayerSelection[] }>({});
    const [totalRivals, setTotalRivals] = useState<number | null>(null);

    useEffect(() => {
        let toSet: { [id: number]: PlayerSelection[] } = {};
        if (!!selectedLeague && !!allData?.players && !!allData.leagues && !!allData.myDetails) {
            const { players, leagues, myDetails } = allData!;
            const league = leagues.find(l => l.id == selectedLeague)!;
            setTotalRivals(league.participants ? league.participants.length - 1 : null);
            players!.forEach(player => {
                const selectionsForPlayer: PlayerSelection[] = [];
                league.participants
                    ?.filter(p => p.userId != myDetails.id)
                    .forEach(participant => {
                        const isSelectedByOtherPlayer = participant.startingXi?.some(p => p === player!.id)
                            || participant.bench?.some(p => p === player!.id);
                        if (isSelectedByOtherPlayer) {
                            selectionsForPlayer.push(new PlayerSelection(participant.playerName!, participant.userId!))
                        }
                    });
                toSet[player!.id!] = selectionsForPlayer;
            });
        }
        setPlayerSelections(toSet);
    }, [selectedLeague, allData])

    if (!allData) {
        return <></>;
    }

    const { players, teams, myTeam } = allData;
    const showRivalSelectionCount = Object.keys(playerSelections).length > 0 && totalRivals != null;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4 flex flex-col">
        <h2 className="text-xl font-semibold mb-4">Players</h2>
        <table>
            <tbody>
                {players.slice(0, 50).map((player, index) => (
                    <tr key={index} className={playerBg(player)}>
                        <td className="font-medium flex justify-between items-center px-2">
                            <div>{player.name} </div>
                            <SelectionIcon player={player} team={myTeam} />
                            {showRivalSelectionCount ? <RivalSelectionCount totalRivals={totalRivals} selections={playerSelections[player!.id!]} /> : <></>}
                        </td>
                        <td className="text-sm">{lookupTeam(player.team!, teams).shortName}</td>
                        <td className="text-gray-500 font-mono text-sm text-right">£{(player.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 font-mono text-sm text-right">{player.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
}

export default Players;