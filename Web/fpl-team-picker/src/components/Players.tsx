import { useContext } from "react";
import { DataContext } from "../App";
import { lookupTeam } from "../helpers/lookups";
import { Position } from "../models/position";

function Players() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }

    const { players, teams } = allData;

    return <div>
        {players.slice(0, 15).map((player, i) => {
            return <div className="flex gap-1" key={`${i}-xi`}>
                <div>{player.name}</div>
                <div>{Position[player.position!]}</div>
                <div>{lookupTeam(player.team!, teams).shortName}</div>
                <div>{player.cost}</div>
                <div>{player.xpNext}</div>
            </div>
        })}
    </div>
}

export default Players;