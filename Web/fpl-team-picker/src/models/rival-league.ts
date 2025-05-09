import { LeagueParticipant, SelectedSquad } from "../helpers/api";

export class RivalTeam {
  constructor(public rival: LeagueParticipant, public team: SelectedSquad) { }
}