import { useContext, useEffect, useState } from "react";
import { DataContext, SelectedLeagueContext } from "../App";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";
import { PlayerSelection } from "../models/player-selection";

function MyTeam() {
    const allData = useContext(DataContext);
    const selectedLeague = useContext(SelectedLeagueContext);
    const [playerSelections, setPlayerSelections] = useState<{ [id: number]: PlayerSelection[] }>({});

    useEffect(() => {
        let toSet = {};
        if (!!selectedLeague && !!allData?.myTeam && !!allData.leagues) {
            const { myTeam, leagues } = allData!;
            myTeam.startingXi!.forEach(player => {
                const league = leagues.find(l => l.id == selectedLeague)!;
                league.participants?.forEach(participant => {
                    //todo return the players for it as well
                })
            });
        }
        setPlayerSelections(toSet);
    }, [selectedLeague, allData])

    if (!allData) {
        return <></>;
    }
    const { myTeam, teams } = allData;

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
                        <td className="font-medium">{player.player!.name}</td>
                        <td className="text-sm">{lookupTeam(player.player!.team!, teams).shortName}</td>
                        <td className="text-gray-500 font-mono text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 font-mono text-sm text-right">{player.player!.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
                <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>bench</td></tr>
                {myTeam.bench!.map((player, index) => (
                    <tr key={index} className={playerBg(player.player!)}>
                        <td className="font-medium">{player.player!.name}</td>
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