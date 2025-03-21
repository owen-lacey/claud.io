import { Player, Team } from "../../helpers/api";
import { lookupTeam } from "../../helpers/lookups";
import { Position } from "../../models/position";

function PlayerRow({ player, teams }: { player: Player, teams: Team[] }) {
    let bg = '';
    switch(+player.position!) {
        case Position.GK:
            bg = 'bg-red-50';
            break;
        case Position.DEF:
            bg = 'bg-yellow-50';
            break;
        case Position.MID:
            bg = 'bg-green-50';
            break;
        case Position.FWD:
            bg = 'bg-blue-50';
            break;
    }

    return <tr className={'h-8 ' + bg}>
        <td className="font-medium">{player.name}</td>
        <td className="text-gray-500 text-sm">{lookupTeam(player.team!, teams).shortName}</td>
        <td className="text-gray-500 text-sm">£{(player.cost! / 10).toFixed(1)}M</td>
        <td className="text-blue-500 text-sm">XP: {player.xpNext!.toFixed(1)}</td>
    </tr>;
}

export default PlayerRow;