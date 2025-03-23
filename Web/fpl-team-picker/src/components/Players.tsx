import { useContext } from "react";
import { DataContext } from "../App";
import { playerBg } from "../helpers/styles";
import { lookupTeam } from "../helpers/lookups";
import SelectionIcon from "./players/SelectionIcon";

function Players() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }

    const { players, teams, myTeam } = allData;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4 flex flex-col">
        <h2 className="text-xl font-semibold mb-4">Players</h2>
        <table>
            <tbody>
                {players.slice(0, 50).map((player, index) => (
                    <tr key={index} className={playerBg(player)}>
                        <td className="font-medium flex items-center">
                            <div>{player.name} </div>
                            <SelectionIcon player={player} team={myTeam}/>
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