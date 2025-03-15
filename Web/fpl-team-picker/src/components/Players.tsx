import { useEffect, useState } from "react";
import { FplApi } from "../helpers/fpl-api";
import { Player, Team } from "../helpers/api";

function Players() {
    const [players, setPlayers] = useState<Player[] | null>(null);

    const loadDetails = async () => {
        const result = await new FplApi().players.playersList();
        setPlayers(result.data);
    }

    useEffect(() => {
        loadDetails();
    }, []);

    if (!players) {
        return <></>;
    }

    return <div>
            {players?.slice(0, 15).map((player, i) => {
                return <div className="flex gap-1" key={`${i}-xi`}>
                    <div>{player?.name}</div>
                    <div>{player?.position}</div>
                    <div>{player?.team}</div>
                    <div>{player?.cost}</div>
                    <div>{player?.xpNext}</div>
                </div>
            })}
    </div>
}

export default Players;