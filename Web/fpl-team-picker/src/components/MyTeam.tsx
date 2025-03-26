import { useContext, useEffect, useState } from "react";
import { DataContext, SelectedLeagueContext } from "../App";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";
import { PlayerSelection } from "../models/player-selection";
import RivalSelectionCount from "./players/RivalSelectionCount";

function MyTeam() {
    const allData = useContext(DataContext);
    const selectedLeague = useContext(SelectedLeagueContext);
    const [playerSelections, setPlayerSelections] = useState<{ [id: number]: PlayerSelection[] }>({});
    const [totalRivals, setTotalRivals] = useState<number | null>(null);

    useEffect(() => {
        let toSet: { [id: number]: PlayerSelection[] } = {};
        if (!!selectedLeague && !!allData?.myTeam && !!allData.leagues && !!allData.myDetails) {
            const { myTeam, leagues, myDetails } = allData!;
            const league = leagues.find(l => l.id == selectedLeague)!;
            setTotalRivals(league.participants ? league.participants.length - 1 : null);
            myTeam.startingXi!.forEach(player => {
                const selectionsForPlayer: PlayerSelection[] = [];
                league.participants
                    ?.filter(p => p.userId != myDetails.id)
                    .forEach(participant => {
                        const isSelectedByOtherPlayer = participant.startingXi?.some(p => p === player.player!.id)
                            || participant.bench?.some(p => p === player.player!.id);
                        if (isSelectedByOtherPlayer) {
                            selectionsForPlayer.push(new PlayerSelection(participant.playerName!, participant.userId!, participant.position!))
                        }
                    });
                toSet[player.player!.id!] = selectionsForPlayer;
            });
            myTeam.bench!.forEach(player => {
                const selectionsForPlayer: PlayerSelection[] = [];
                league.participants
                    ?.filter(p => p.userId != myDetails.id)
                    .forEach(participant => {
                        const isSelectedByOtherPlayer = participant.startingXi?.some(p => p === player.player!.id)
                            || participant.bench?.some(p => p === player.player!.id);
                        if (isSelectedByOtherPlayer) {
                            selectionsForPlayer.push(new PlayerSelection(participant.playerName!, participant.userId!, participant.position!))
                        }
                    });
                toSet[player.player!.id!] = selectionsForPlayer;
            });
        }
        setPlayerSelections(toSet);
    }, [selectedLeague, allData])

    if (!allData) {
        return <></>;
    }
    const { myTeam, teams } = allData;

    const showRivalSelectionCount = Object.keys(playerSelections).length > 0 && totalRivals != null;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4 flex flex-col">
        <div className="flex gap-1 justify-around mb-4">
            <div className="w-19 rounded-md bg-blue-950 p-2 text-gray-300 font-mono flex flex-col items-center justify-center">
                <div className="text-lg">£{(myTeam.bank! / 10).toFixed(1)}</div>
                <div className="text-xs uppercase">bank</div>
            </div>
            <div className="w-19 rounded-md bg-blue-950 p-2 text-gray-300 font-mono flex flex-col items-center justify-center">
                <div className="text-lg">£{(myTeam.budget! / 10).toFixed(1)}</div>
                <div className="text-xs uppercase">value</div>
            </div>
            <div className="w-19 rounded-md bg-blue-950 p-2 text-gray-300 font-mono flex flex-col items-center justify-center">
                <div className="text-lg">{myTeam.freeTransfers}</div>
                <div className="text-xs uppercase">ft</div>
            </div>
        </div>
        <table>
            <tbody className="">
                <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>XI</td></tr>
                {myTeam.startingXi!.map((player, index) => (
                    <tr key={index} className={playerBg(player.player!)}>
                        <td className="font-medium flex justify-between items-center px-2">
                            <div>{player.player!.name}</div>
                            {showRivalSelectionCount ? <RivalSelectionCount totalRivals={totalRivals} selections={playerSelections[player.player!.id!]} /> : <></>}
                        </td>
                        <td className="text-sm">{lookupTeam(player.player!.team!, teams).shortName}</td>
                        <td className="text-gray-500 font-mono text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 font-mono text-sm text-right">{player.player!.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
                <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>bench</td></tr>
                {myTeam.bench!.map((player, index) => (
                    <tr key={index} className={playerBg(player.player!)}>
                        <td className="font-medium flex justify-between items-center px-2">
                            <div>{player.player!.name}</div>
                            {showRivalSelectionCount ? <RivalSelectionCount totalRivals={totalRivals} selections={playerSelections[player.player!.id!]} /> : <></>}
                        </td>
                        <td className="text-sm">{lookupTeam(player.player!.team!, teams).shortName}</td>
                        <td className="text-gray-500 font-mono text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 font-mono text-sm text-right">{player.player!.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
        <div className="flex justify-end py-4 text-blue-500 text-sm">
            <div>XP (bench boost):&nbsp;</div>
            <div className="font-mono">{myTeam.predictedPoints?.toFixed(1)} ({myTeam.benchBoostPredictedPoints?.toFixed(1)})</div>
        </div>
    </div>
}

export default MyTeam;