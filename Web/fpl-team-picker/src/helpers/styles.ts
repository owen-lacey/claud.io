import { Position } from "../models/position";
import { Player } from "./api";

export function playerBg(player: Player) {

    let bg = '';
    switch (+player.position!) {
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

    return bg;
}