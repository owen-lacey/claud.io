import { Team } from "./api";

export function lookupTeam(teamId: number, teams: Team[]): Team {
    return teams.find(t => t.id == teamId)!;
}