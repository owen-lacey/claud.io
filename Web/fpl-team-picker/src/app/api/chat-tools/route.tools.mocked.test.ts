/**
 * @jest-environment node
 */

// Deterministic unit test for /api/chat-tools tools-mode contract by mocking the AI stream.
// Verifies f: (tool call) and d: (tool result) frames without hitting the real model.

jest.mock('ai', () => {
  return {
    streamText: () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          // Emit a tool-call frame (pretend the model called list_players)
          const toolCall = 'f:' + JSON.stringify({ name: 'list_players', arguments: { position: 3, limit: 1 } }) + '\n';
          controller.enqueue(encoder.encode(toolCall));

          // Emit a deterministic tool result payload
          const payload = { items: [{ id: 10, name: 'Player A', position: 3, team: 1, cost: 55, xp: 6.2 }] };
          const dataFrame = 'd:' + JSON.stringify(payload) + '\n';
          controller.enqueue(encoder.encode(dataFrame));

          controller.close();
        },
      });

      return {
        toDataStreamResponse() {
          return new Response(stream, {
            status: 200,
            headers: { 'content-type': 'text/plain; charset=utf-8' },
          });
        },
      } as any;
    },
  };
});

import { POST } from './route';

describe('/api/chat-tools (tools mode) — mocked stream', () => {
  it('emits tool-call and tool-result frames', async () => {
    const req = new Request('http://localhost:3000/api/chat-tools', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', pl_profile: 'test-token' },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: 'Test' },
          { role: 'user', content: 'List midfielders' },
        ],
      }),
    });

    const res = await POST(req as any);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/plain');

    const reader = res.body?.getReader();
    expect(reader).toBeDefined();

    let output = '';
    if (reader) {
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) output += decoder.decode(value);
      }
    }

    expect(output).toMatch(/(^|\n)f:\{/); // tool-call frame exists
    expect(output).toMatch(/(^|\n)d:\{/); // tool-result frame exists

    const dLine = output.split('\n').find((l) => l.startsWith('d:'));
    expect(dLine).toBeDefined();
    if (dLine) {
      const payload = JSON.parse(dLine.slice(2));
      expect(payload).toHaveProperty('items');
      expect(Array.isArray(payload.items)).toBe(true);
    }
  });
});
