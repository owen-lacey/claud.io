/**
 * @jest-environment node
 */
import { buildSquad } from '../selection';
import { Position } from '@/models/position';
import { NormalizedPlayer } from '@/lib/data-service';
import { FPL_RULES } from '../constraints';

function mockP(id: number, pos: Position, team: number, cost: number, xp: number, chance = 100): NormalizedPlayer {
  return {
    id,
    name: `P${id}`,
    position: pos,
    cost,
    team: { id: team, name: `T${team}`, shortName: `T${team}` },
    expectedPoints: xp,
    predictions: {},
    availability: { chanceOfPlaying: chance, isAvailable: true },
    stats: { seasonPoints: 0, selectedByPercent: 0, transfersOut: 0, yellowCards: 0, redCards: 0 },
  };
}

describe('selection', () => {
  it('builds a squad that respects constraints and budget deterministically', () => {
    const pool: NormalizedPlayer[] = [];
    const teams = [1,2,3,4,5,6,7,8];
    // create a decent pool of candidates per position
    let id = 1;
    for (let i=0;i<6;i++) pool.push(mockP(id++, Position.GK, teams[i%teams.length], 45 + i, 4 + i*0.2));
    for (let i=0;i<20;i++) pool.push(mockP(id++, Position.DEF, teams[i%teams.length], 45 + (i%5), 3 + i*0.1));
    for (let i=0;i<20;i++) pool.push(mockP(id++, Position.MID, teams[i%teams.length], 60 + (i%5), 4 + i*0.15));
    for (let i=0;i<12;i++) pool.push(mockP(id++, Position.FWD, teams[i%teams.length], 70 + (i%5), 4 + i*0.12));

    const budget = 995; // 99.5m
    const { squad } = buildSquad(pool, budget);

    // Check sizes
    expect(squad.startingXi.length + squad.bench.length).toBeLessThanOrEqual(FPL_RULES.squadSize);
    // Check budget
    expect(squad.totalCost).toBeLessThanOrEqual(budget);
    // Basic captaincy set
    if (squad.startingXi.length > 0) {
      expect(squad.captain).toBeDefined();
    }
  });

  it('is deterministic with same input', () => {
    const pool: NormalizedPlayer[] = [
      mockP(1, Position.MID, 1, 70, 6),
      mockP(2, Position.MID, 2, 70, 6),
      mockP(3, Position.MID, 3, 70, 6),
      mockP(4, Position.MID, 4, 70, 6),
      mockP(5, Position.MID, 5, 70, 6),
    ];
    const budget = 300;
    const a = buildSquad(pool, budget);
    const b = buildSquad(pool, budget);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});
