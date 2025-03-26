import { Player, Team } from "./api";

export function lookupTeam(teamId: number, teams: Team[]): Team {
    return teams.find(t => t.id == teamId)!;
}

export function lookupPlayer(playerId: number, players: Player[]): Player {
    return players.find(p => p.id == playerId)!;
}