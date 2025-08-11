import { google } from "@ai-sdk/google";
import { streamText } from "ai";

export const maxDuration = 30;

export async function POST(req: Request) {
  // Parse body safely and normalize messages to plain text
  let body: any = null;
  try {
    body = await req.json();
  } catch {
    return new Response(`3:${JSON.stringify('[chat] Invalid JSON payload')}\n`, {
      status: 200,
      headers: { 'content-type': 'text/plain; charset=utf-8' },
    });
  }

  const rawMessages = Array.isArray(body?.messages) ? body.messages : [];
  const normalized = rawMessages.map((m: any) => {
    const c = m?.content;
    let text = '';
    if (typeof c === 'string') text = c;
    else if (Array.isArray(c)) {
      text = c.map((p: any) => (typeof p === 'string' ? p : p?.text ?? '')).filter(Boolean).join(' ');
    }
    return { role: m?.role ?? 'user', content: text } as { role: string; content: string };
  });
  // Filter to roles accepted by AI SDK and drop empty content
  const allowedRoles = new Set(['system', 'user', 'assistant']);
  const messages = normalized.filter((m: { role: string; content: string }) => allowedRoles.has(m.role) && typeof m.content === 'string' && m.content.trim().length > 0);

  try {
    const result = streamText({
      model: google("gemini-2.0-flash"),
      messages,
    });
    return result.toDataStreamResponse();
  } catch (err: any) {
    const status = err?.status ?? err?.response?.status;
    const code = err?.code ?? err?.response?.data?.error?.code;
    const msg = `[chat] ${err?.message ?? 'Streaming failed.'}${status ? ` (status=${status})` : ''}${code ? ` (code=${code})` : ''}`;
    console.error('[chat] stream error', { status, code, error: err });
    return new Response(`3:${JSON.stringify(msg)}\n`, {
      status: 200,
      headers: { 'content-type': 'text/plain; charset=utf-8' },
    });
  }
}