/**
 * @jest-environment node
 */
import { Position } from '@/models/position';

jest.mock('@/lib/data-service', () => {
  function mk(id: number, pos: any, team: number, cost: number, xp: number) {
    return {
      id,
      name: `P${id}`,
      position: pos,
      cost,
      team: { id: team, name: `T${team}`, shortName: `T${team}` },
      expectedPoints: xp,
      predictions: {},
      availability: { chanceOfPlaying: 100, isAvailable: true },
      stats: { seasonPoints: 0, selectedByPercent: 0, transfersOut: 0, yellowCards: 0, redCards: 0 },
    };
  }
  const players = [
    mk(1, Position.GK, 1, 45, 5), mk(2, Position.GK, 2, 45, 4),
    mk(3, Position.DEF, 3, 45, 4), mk(4, Position.DEF, 4, 45, 4), mk(5, Position.DEF, 5, 45, 3), mk(6, Position.DEF, 6, 45, 3), mk(7, Position.DEF, 7, 45, 3),
    mk(8, Position.MID, 3, 60, 5), mk(9, Position.MID, 4, 60, 5), mk(10, Position.MID, 5, 60, 4), mk(11, Position.MID, 6, 60, 4), mk(12, Position.MID, 7, 60, 4),
    mk(13, Position.FWD, 3, 70, 5), mk(14, Position.FWD, 4, 70, 4), mk(15, Position.FWD, 5, 70, 4),
  ];
  return {
    dataService: {
      async getSelectionContext() {
        return {
          players,
          teams: [],
          myTeam: { freeTransfers: 1, bank: 0, budget: 1000, squad: null },
          user: { id: 1, name: 'Test' },
        } as any;
      },
    },
  };
});

import { POST } from './route';

describe('/api/tools/build-squad', () => {
  it('returns 200 and a squad JSON with explanation', async () => {
    const req = new Request('http://localhost:3000/api/tools/build-squad', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ budget: 1000 }),
    });

    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json).toHaveProperty('squad');
    expect(json).toHaveProperty('explanation');
    expect(json.squad.startingXi.length + json.squad.bench.length).toBeGreaterThan(0);
  }, 10000);
});
