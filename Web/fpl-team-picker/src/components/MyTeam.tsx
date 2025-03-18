import { useContext } from "react";
import { DataContext } from "../App";
import { lookupTeam } from "../helpers/lookups";
import { Position } from "../models/position";

function MyTeam() {
    const allData = useContext(DataContext);

    if (!allData) {
        return <></>;
    }
    const { myTeam, teams } = allData;

    return <div>
        <div className="flex gap-1">
            <div>{myTeam.bank}</div>
            <div>{myTeam.budget}</div>
            <div>{myTeam.freeTransfers}</div>
        </div>
        <div>
            {myTeam.startingXi!.map((player, i) => {
                return <div className="flex gap-1" key={`${i}-xi`}>
                    <div>{player.player!.name}</div>
                    <div>{Position[player.player!.position!]}</div>
                    <div>{lookupTeam(player.player!.team!, teams).shortName}</div>
                    <div>{player.player!.cost}</div>
                    <div>{player.player!.xpNext}</div>
                </div>
            })}
        </div>
        <div>
            {myTeam.bench!.map((player, i) => {
                return <div className="flex" key={`${i}-bench`}>
                    <div>{player.player!.name}</div>
                    <div>{Position[player.player!.position!]}</div>
                    <div>{lookupTeam(player.player!.team!, teams).shortName}</div>
                    <div>{player.player!.cost}</div>
                </div>
            })}
        </div>
    </div>
}

export default MyTeam;