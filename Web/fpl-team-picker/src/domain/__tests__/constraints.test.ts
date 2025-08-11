/**
 * @jest-environment node
 */

import { addToCounts, emptyCounts, FPL_RULES, isValidFinalSquad, isValidPartialSquad, respectsMaxPerClub, respectsPositionQuotas, validCaptaincy, withinBudget } from '../constraints';
import { Position } from '@/models/position';
import { NormalizedPlayer } from '@/lib/data-service';

function mockPlayer(id: number, position: Position, teamId: number): NormalizedPlayer {
  return {
    id,
    name: `P${id}`,
    position,
    cost: 50,
    team: { id: teamId, name: `T${teamId}`, shortName: `T${teamId}` },
    expectedPoints: 3,
    predictions: {},
    availability: { chanceOfPlaying: 100, isAvailable: true },
    stats: { seasonPoints: 0, selectedByPercent: 0, transfersOut: 0, yellowCards: 0, redCards: 0 },
  };
}

describe('constraints', () => {
  it('should accumulate position counts and per-club limits', () => {
    let counts = emptyCounts();
    counts = addToCounts(counts, mockPlayer(1, Position.GK, 1));
    counts = addToCounts(counts, mockPlayer(2, Position.DEF, 1));
    counts = addToCounts(counts, mockPlayer(3, Position.MID, 2));
    counts = addToCounts(counts, mockPlayer(4, Position.FWD, 2));

    expect(counts.total).toBe(4);
    expect(counts.GK).toBe(1);
    expect(counts.DEF).toBe(1);
    expect(counts.MID).toBe(1);
    expect(counts.FWD).toBe(1);
    expect(counts.perTeam[1]).toBe(2);
    expect(counts.perTeam[2]).toBe(2);
  });

  it('respects position quotas for partial squads', () => {
    let counts = emptyCounts();
    for (let i = 0; i < FPL_RULES.positionQuotas.GK; i++) counts = addToCounts(counts, mockPlayer(10 + i, Position.GK, 1));
    expect(respectsPositionQuotas(counts)).toBe(true);
    // Adding one more GK breaks quota
    counts = addToCounts(counts, mockPlayer(20, Position.GK, 2));
    expect(respectsPositionQuotas(counts)).toBe(false);
  });

  it('respects max per club', () => {
    let counts = emptyCounts();
    counts = addToCounts(counts, mockPlayer(1, Position.DEF, 1));
    counts = addToCounts(counts, mockPlayer(2, Position.MID, 1));
    counts = addToCounts(counts, mockPlayer(3, Position.FWD, 1));
    expect(respectsMaxPerClub(counts)).toBe(true);
    counts = addToCounts(counts, mockPlayer(4, Position.GK, 1));
    expect(respectsMaxPerClub(counts)).toBe(false);
  });

  it('validates budget', () => {
    expect(withinBudget(1000, 0)).toBe(true);
    expect(withinBudget(1001, 0)).toBe(false);
    expect(withinBudget(900, -1)).toBe(false);
  });

  it('validates final squad composition', () => {
    let counts = emptyCounts();
    // Distribute exactly 3 players per club across 5 clubs to respect max-per-club=3
    const GKTeams = [1, 2];
    const DEFTeams = [3, 4, 5, 1, 2];
    const MIDTeams = [3, 4, 5, 1, 2];
    const FWDTeams = [3, 4, 5];

    GKTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(100 + i, Position.GK, t)); });
    DEFTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(200 + i, Position.DEF, t)); });
    MIDTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(300 + i, Position.MID, t)); });
    FWDTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(400 + i, Position.FWD, t)); });

    expect(isValidPartialSquad(counts)).toBe(true);
    expect(isValidFinalSquad(counts, 980, 20)).toBe(true);
  });

  it('validates final squad at exact budget cap', () => {
    let counts = emptyCounts();
    const GKTeams = [1, 2];
    const DEFTeams = [3, 4, 5, 1, 2];
    const MIDTeams = [3, 4, 5, 1, 2];
    const FWDTeams = [3, 4, 5];

    GKTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(500 + i, Position.GK, t)); });
    DEFTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(600 + i, Position.DEF, t)); });
    MIDTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(700 + i, Position.MID, t)); });
    FWDTeams.forEach((t, i) => { counts = addToCounts(counts, mockPlayer(800 + i, Position.FWD, t)); });

    expect(isValidFinalSquad(counts, FPL_RULES.budgetCap, 0)).toBe(true);
  });

  it('allows exactly 3 per club in partial squad', () => {
    let counts = emptyCounts();
    // exactly 3 from club 1
    counts = addToCounts(counts, mockPlayer(900, Position.GK, 1));
    counts = addToCounts(counts, mockPlayer(901, Position.DEF, 1));
    counts = addToCounts(counts, mockPlayer(902, Position.MID, 1));
    // some more players from other clubs within quotas
    counts = addToCounts(counts, mockPlayer(903, Position.DEF, 2));
    counts = addToCounts(counts, mockPlayer(904, Position.MID, 3));
    counts = addToCounts(counts, mockPlayer(905, Position.FWD, 4));

    expect(isValidPartialSquad(counts)).toBe(true);

    // adding a 4th from club 1 breaks per-club limit
    counts = addToCounts(counts, mockPlayer(906, Position.DEF, 1));
    expect(isValidPartialSquad(counts)).toBe(false);
  });

  it('valid captaincy placeholder', () => {
    const squad = [mockPlayer(1, Position.MID, 1), mockPlayer(2, Position.DEF, 2)];
    expect(validCaptaincy(1, 2, squad)).toBe(true);
    expect(validCaptaincy(1, 1, squad)).toBe(false);
    expect(validCaptaincy(1, 3, squad)).toBe(false);
    expect(validCaptaincy(null, null, squad)).toBe(true);
  });
});
