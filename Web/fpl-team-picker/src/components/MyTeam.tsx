import { useContext } from "react";
import { DataContext } from "../App";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";

function MyTeam() {
    const allData = useContext(DataContext);

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
                    <tr key={index} className={'h-8 ' + playerBg(player.player!)}>
                        <td className="font-medium">{player.player!.name}</td>
                        <td className="text-gray-500 text-sm">{lookupTeam(player.player!.team!, teams).shortName}</td>
                        <td className="text-gray-500 text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 text-sm text-right">XP: {player.player!.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
                <tr className="font-mono text-sm text-gray-400 uppercase text-center"><td className="p-2" key={0} colSpan={99}>bench</td></tr>
                {myTeam.bench!.map((player, index) => (
                    <tr key={index} className={'h-8 ' + playerBg(player.player!)}>
                        <td className="font-medium">{player.player!.name}</td>
                        <td className="text-gray-500 text-sm">{lookupTeam(player.player!.team!, teams).shortName}</td>
                        <td className="text-gray-500 text-sm text-right">£{(player.player!.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 text-sm text-right">XP: {player.player!.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
}

export default MyTeam;