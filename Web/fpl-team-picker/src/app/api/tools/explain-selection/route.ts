import { NextRequest } from 'next/server';
import { explainSelectionTool } from '@/app/api/chat-tools/tools/explain_selection';

export const maxDuration = 30;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const tool = explainSelectionTool();
    const result = await tool.execute(body);
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
