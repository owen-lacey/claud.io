import { LeagueParticipant, SelectedTeam } from "../helpers/api";

export class RivalTeam {
  constructor(public rival: LeagueParticipant, public team: SelectedTeam) { }
}