import { League, MyTeam, Player, Team, User } from "../helpers/api";

export class AllData {
  constructor(
    public myTeam: MyTeam,
    public players: Player[],
    public teams: Team[],
    public leagues: League[],
    public myDetails: User) {
  }
}