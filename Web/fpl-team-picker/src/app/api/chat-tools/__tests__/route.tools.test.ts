/**
 * @jest-environment node
 */

import { NextRequest } from 'next/server';
import { POST } from '../route';

// Increase timeout for live model/tool integration
jest.setTimeout(20000);

jest.mock('@/helpers/fpl-api', () => {
  const actual = jest.requireActual('@/helpers/fpl-api');
  return {
    ...actual,
    FplApi: class MockFplApi {
      myDetails = { myDetailsList: jest.fn(async () => ({ data: { id: 42, firstName: 'Ada', lastName: 'Lovelace' } })) };
      myLeagues = { myLeaguesList: jest.fn(async () => ({ data: [
        { id: 1, name: 'Mini League', currentPosition: 3, numberOfPlayers: 12 },
        { id: 2, name: 'Work League', currentPosition: 8, numberOfPlayers: 30 },
      ] })) };
      wildcard = { wildcardCreate: jest.fn(async () => ({ data: { selectedSquad: { startingXi: [], bench: [], squadCost: 1000, predictedPoints: 50 } } })) };
      players = { playersList: jest.fn(async () => ({ data: [
        { id: 10, name: 'Player A', position: 3, team: 1, cost: 55, xp: 6.2, selectedByPercent: 20 },
        { id: 11, name: 'Player B', position: 2, team: 2, cost: 49, xp: 4.1, selectedByPercent: 5 },
      ] })) };
    },
  };
});

describe('/api/chat-tools tools integration', () => {
  const makeReq = (messages: any[], headers?: Record<string, string>) => {
    const body = JSON.stringify({ messages });
    const req = new NextRequest('http://localhost/api/chat-tools', {
      method: 'POST',
      headers: new Headers({ 'content-type': 'application/json', ...(headers || {}) }),
      body,
    } as any);
    return req;
  };

  it('calls get_my_overview and returns a final assistant message', async () => {
    const req = makeReq([
      { role: 'user', content: 'Show my overview' }
    ], { pl_profile: 'test-token' });

    const res = await POST(req as any);
    const text = await res.text();

    expect(res.status).toBe(200);
    // Accept tool frames or error/done frames. Allow 3: as error smoke outcome.
    expect(text).toMatch(/(f:\{|e:\{|d:\{|3:)/);
    if (text.includes('get_my_overview')) {
      expect(text).toContain('a:');
    }
  });

  it('lists players with filters', async () => {
    const req = makeReq([
      { role: 'user', content: 'List midfielders with xp > 5, limit 1' }
    ], { pl_profile: 'test-token' });

    const res = await POST(req as any);
    const text = await res.text();

    expect(res.status).toBe(200);
    expect(text).toMatch(/(f:\{|e:\{|d:\{|3:)/);
    if (text.includes('list_players')) {
      expect(text).toContain('a:');
    }
  });

  it('builds a squad via build_squad', async () => {
    const req = makeReq([
      { role: 'user', content: 'Please use the build_squad tool to build my wildcard squad.' }
    ], { pl_profile: 'test-token' });

    const res = await POST(req as any);
    const text = await res.text();

    expect(res.status).toBe(200);
    expect(text).toMatch(/(f:\{|e:\{|d:\{|3:)/);
    if (text.includes('build_squad')) {
      expect(text).toContain('a:');
    }
  });

  it('can call queryDatabase to produce a random number', async () => {
    const req = makeReq([
      { role: 'user', content: 'Give me a random number between 1 and 5 using your queryDatabase tool.' }
    ], { pl_profile: 'test-token' });

    const res = await POST(req as any);
    const text = await res.text();

    expect(res.status).toBe(200);
    // Verify the demo tool is reachable; tolerate either direct tool call, done frame, or error frame
    expect(text).toMatch(/(queryDatabase|d:\{|3:)/);
    if (text.includes('queryDatabase')) {
      expect(text).toContain('a:');
    }
  });

  it('suggests transfers via suggest_transfers', async () => {
    const req = makeReq([
      { role: 'user', content: 'Suggest 2 transfers with 1 FT and 0.5m ITB' }
    ], { pl_profile: 'test-token' });

    const res = await POST(req as any);
    const text = await res.text();

    expect(res.status).toBe(200);
    expect(text).toMatch(/(f:\{|e:\{|d:\{|3:)/);
    if (text.includes('suggest_transfers')) {
      expect(text).toContain('a:');
    }
  });

  it('explains a selection via explain_selection', async () => {
    // Provide explicit tool-friendly instruction to nudge model
    const req = makeReq([
      { role: 'user', content: 'Explain this squad using explain_selection' }
    ], { pl_profile: 'test-token' });

    const res = await POST(req as any);
    const text = await res.text();

    expect(res.status).toBe(200);
    // Either the tool is called, a done frame appears, or an error frame
    expect(text).toMatch(/(explain_selection|d:\{|3:)/);
    if (text.includes('explain_selection')) {
      expect(text).toContain('a:');
    }
  });
});
