/**
 * @jest-environment node
 */
import { scorePlayer, comparePlayers, defaultScoringOptions } from '../scoring';
import { Position } from '@/models/position';
import type { NormalizedPlayer } from '@/lib/data-service';

function mockP(id: number, xp: number, cost: number, chance = 100): NormalizedPlayer {
  return {
    id,
    name: `P${id}`,
    position: Position.MID,
    cost,
    team: { id: id % 5, name: 'T', shortName: 'T' },
    expectedPoints: xp,
    predictions: {},
    availability: { chanceOfPlaying: chance, isAvailable: true },
    stats: { seasonPoints: 0, selectedByPercent: 0, transfersOut: 0, yellowCards: 0, redCards: 0 },
  };
}

describe('scoring', () => {
  it('scores primarily by expected points and availability, not price by default', () => {
    const a = mockP(1, 6, 120, 100);
    const b = mockP(2, 5, 50, 100);
    const sa = scorePlayer(a);
    const sb = scorePlayer(b);
    expect(sa).toBeGreaterThan(sb);
  });

  it('deterministic compare with tie-breakers', () => {
    const a = mockP(10, 5, 100, 90);
    const b = mockP(11, 5, 100, 95);
    // b has higher availability -> comes first
    expect(comparePlayers(a, b)).toBeGreaterThan(0);
  });

  it('can include price when configured', () => {
    const a = mockP(20, 5, 100, 100);
    const b = mockP(21, 5, 50, 100);
    const opts = { ...defaultScoringOptions, priceWeight: 1 };
    // with priceWeight, b (cheaper) gets higher score when xp equal
    expect(comparePlayers(a, b, opts)).toBeGreaterThan(0);
  });
});
