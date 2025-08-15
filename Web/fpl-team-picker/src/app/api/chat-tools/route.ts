import { google } from "@ai-sdk/google";
import { streamText } from "ai";
import { FplApi } from "@/helpers/fpl-api";
import { buildSquadTool } from "./tools/build_squad";
import { getMyOverviewTool } from "./tools/get_my_overview";
import { listPlayersTool } from "./tools/list_players";
import { listTeamsTool } from "./tools/list_teams";
import { listFixturesTool } from "./tools/list_fixtures";

export const maxDuration = 30;

export async function POST(req: Request) {
  // Parse body safely and normalize messages to plain text
  let body: any = null;
  try {
    body = await req.json();
  } catch {
    return new Response(`3:${JSON.stringify('[chat-tools] Invalid JSON payload')}\n`, {
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

  // Extract auth token for backend (prefer pl_profile, fallback to Bearer token)
  const authHeader = req.headers.get('authorization') || undefined;
  const plProfileHeader = req.headers.get('pl_profile') || undefined;
  const bearerToken = authHeader?.startsWith('Bearer ') ? authHeader.substring('Bearer '.length) : authHeader;
  const token = plProfileHeader || bearerToken;
  console.log('[chat-tools] token info', token ? `present(len=${token.length})` : 'absent');
  const api = new FplApi(token);

  try {
    console.log('[chat-tools] Starting streamText with messages:', messages.length);
    const result = streamText({
      model: google('gemini-2.0-flash'),
      system: 'You are an FPL assistant. Use get_my_overview when the user asks about account or leagues; use build_squad for team building. Use list_players for quick lists of players. Use list_teams for team information. Use list_fixtures for fixture information and schedules. After any tool call completes, always produce a short final assistant message summarizing the results.',
      messages,
      maxSteps: 3,
      tools: {
        build_squad: buildSquadTool(api, token),
        get_my_overview: getMyOverviewTool(api, token),
        list_players: listPlayersTool(api),
        list_teams: listTeamsTool(api),
        list_fixtures: listFixturesTool(api),
      }
    });

    console.log('[chat-tools] streamText created, returning response');
    return result.toDataStreamResponse();
  } catch (err: any) {
    const status = err?.status ?? err?.response?.status;
    const code = err?.code ?? err?.response?.data?.error?.code;
    const msg = `[chat-tools] ${err?.message ?? 'Streaming failed.'}${status ? ` (status=${status})` : ''}${code ? ` (code=${code})` : ''}`;
    console.error('[chat-tools] stream error', { status, code, error: err });
    return new Response(`3:${JSON.stringify(msg)}\n`, {
      status: 200,
      headers: { 'content-type': 'text/plain; charset=utf-8' },
    });
  }
}
