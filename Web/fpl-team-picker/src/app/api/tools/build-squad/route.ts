import { NextRequest } from 'next/server';
import { dataService } from '@/lib/data-service';
import { buildSquad } from '@/domain/selection';

export const maxDuration = 30;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const budget = typeof body?.budget === 'number' ? body.budget : undefined;

    const { players, myTeam } = await dataService.getSelectionContext();
    const effectiveBudget = budget ?? (myTeam?.budget ?? 1000);

    const { squad, explanation } = buildSquad(players, effectiveBudget);

    return new Response(JSON.stringify({ squad, explanation }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  } catch (err: any) {
    const message = err?.message || 'Unknown error';
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { 'content-type': 'application/json' },
    });
  }
}
