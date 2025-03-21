import { League, Player, SelectedTeam, Team } from "../helpers/api";

export class AllData {
    constructor(
        public myTeam: SelectedTeam, 
        public players: Player[], 
        public teams: Team[],
        public leagues: League[]) {
    }
}