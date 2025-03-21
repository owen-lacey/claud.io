import { useContext } from "react";
import { DataContext } from "../App";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";

function Players() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }

    const { players, teams } = allData;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4 flex flex-col">
        <h2 className="text-xl font-semibold mb-4">Players</h2>
        <table>
            <tbody>
                {players.slice(0, 19).map((player, index) => (
                    <tr key={index} className={'h-8 ' + playerBg(player)}>
                        <td className="font-medium">{player.name}</td>
                        <td className="text-gray-500 text-sm">{lookupTeam(player.team!, teams).shortName}</td>
                        <td className="text-gray-500 text-sm text-right">£{(player.cost! / 10).toFixed(1)}</td>
                        <td className="text-blue-500 text-sm text-right">XP: {player.xpNext!.toFixed(1)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
}

export default Players;