import { useContext } from "react";
import { DataContext } from "../App";
import PlayerRow from "./players/PlayerRow";

function MyTeam() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }
    const { myTeam, teams } = allData;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4 flex flex-col">
        <div className="flex gap-1 justify-around mb-4">
            <div className="w-19 rounded-md bg-blue-900 p-4 text-white font-mono flex flex-col items-center">
                <div className="text-lg">£{(myTeam.bank! / 10).toFixed(1)}</div>
                <div className="text-xs uppercase">bank</div>
            </div>
            <div className="w-19 rounded-md bg-blue-900 p-4 text-white font-mono flex flex-col items-center">
                <div className="text-lg">£{(myTeam.budget! / 10).toFixed(1)}</div>
                <div className="text-xs uppercase">value</div>
            </div>
            <div className="w-19 rounded-md bg-blue-900 p-4 text-white font-mono flex flex-col items-center">
                <div className="text-lg">{myTeam.freeTransfers}</div>
                <div className="text-xs uppercase">ft</div>
            </div>
        </div>
        <table>
            <tbody className="">
            <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>XI</td></tr>
                {myTeam.startingXi!.map((player, index) => (
                    <PlayerRow key={index} player={player.player!} teams={teams} />
                ))}
                <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>bench</td></tr>
                {myTeam.bench!.map((player, index) => (
                    <PlayerRow key={index} player={player.player!} teams={teams} />
                ))}
            </tbody>
        </table>
    </div>
}

export default MyTeam;