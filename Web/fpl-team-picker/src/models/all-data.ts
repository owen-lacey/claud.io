import { Player, SelectedTeam, Team } from "../helpers/api";

export class AllData {
    constructor(public myTeam: SelectedTeam, public players: Player[], public teams: Team[]) {
    }
}