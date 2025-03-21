import { useContext } from "react";
import { DataContext } from "../App";

function Leagues() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }
    const { leagues } = allData;

    return <div className="bg-white border border-gray-300 shadow-lg rounded-lg p-4">
        <h2 className="text-xl font-semibold mb-4">My Leagues</h2>
        <table>
            <thead>
                <tr className="font-mono uppercase text-sm text-gray-400 text-left">
                    <th className="px-2">Name</th>
                    <th className="px-2">Pos.</th>
                    <th className="px-2"># players</th>
                </tr>
            </thead>
            <tbody>
                {leagues.map((league, i) => {
                    return <tr key={i}>
                        <td className="px-2">{league!.name}</td>
                        <td className="px-2 text-right">{league.currentPosition}</td>
                        <td className="px-2 text-right">{league.numberOfPlayers}</td>
                    </tr>
                })}
            </tbody>
        </table>
    </div>
}

export default Leagues;