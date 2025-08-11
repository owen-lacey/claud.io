import { NextRequest } from 'next/server';
import { suggestTransfersTool } from '@/app/api/chat-tools/tools/suggest_transfers';
import { FplApi } from '@/helpers/fpl-api';

export const maxDuration = 30;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const { ft, itb, gw } = body || {};

    const bearer = req.headers.get('authorization') || '';
    const token = (req.headers.get('pl_profile') || bearer.replace(/^Bearer\s+/i, '')) || undefined;

    const api = new FplApi(token);
    const tool = suggestTransfersTool(api);
    const result = await tool.execute({ ft, itb, gw });

    return new Response(JSON.stringify(result), {
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
