import { useEffect, useState } from "react";
import { FplApi } from "../helpers/fpl-api";
import { Team } from "../helpers/api";

function MyTeam() {
    const [team, setTeam] = useState<Team | null>(null);

    const loadDetails = async () => {
        const result = await new FplApi().team.teamList();
        setTeam(result.data);
    }

    useEffect(() => {
        loadDetails();
    }, []);

    if (!team) {
        return <></>;
    }

    return <div>
        <div className="flex gap-1">
            <div>{team.bank}</div>
            <div>{team.budget}</div>
            <div>{team.freeTransfers}</div>
        </div>
        <div>
            {team.startingXi?.map((player, i) => {
                return <div className="flex gap-1" key={`${i}-xi`}>
                    <div>{player.player?.name}</div>
                    <div>{player.player?.position}</div>
                    <div>{player.player?.team}</div>
                    <div>{player.player?.cost}</div>
                    <div>{player.player?.xpNext}</div>
                </div>
            })}
        </div>
        <div>
            {team.bench?.map((player, i) => {
                return <div className="flex" key={`${i}-bench`}>
                    <div>{player.player?.name}</div>
                    <div>{player.player?.position}</div>
                    <div>{player.player?.team}</div>
                    <div>{player.player?.cost}</div>
                </div>
            })}
        </div>
    </div>
}

export default MyTeam;