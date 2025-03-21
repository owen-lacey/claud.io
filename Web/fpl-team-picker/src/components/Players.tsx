import { useContext } from "react";
import { DataContext } from "../App";
import PlayerRow from "./players/PlayerRow";

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
                    <PlayerRow key={index} player={player} teams={teams} />
                ))}
            </tbody>
        </table>
    </div>
}

export default Players;