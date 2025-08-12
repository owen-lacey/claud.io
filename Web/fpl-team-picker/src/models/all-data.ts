import { Fixture, League, MyTeam, Player, Team, User } from "../helpers/api";
import { ApiResult } from "./api-result";

export class AllData {
  constructor(
    public myTeam: ApiResult<MyTeam>,
    public players: ApiResult<Player[]>,
    public teams: ApiResult<Team[]>,
    public leagues: ApiResult<League[]>,
    public myDetails: ApiResult<User>,
    public fixtures: ApiResult<Fixture[]>) {
  }
}