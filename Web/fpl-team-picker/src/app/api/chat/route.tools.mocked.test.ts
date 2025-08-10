/**
 * @jest-environment node
 */

// Deterministic unit test for tools-mode contract by mocking the AI stream.
// Verifies f: (tool call) and d: (tool result) frames without hitting the real model.

jest.mock('ai', () => {
  return {
    streamText: (opts: any) => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          // Emit a tool-call frame
          const toolCall = 'f:' + JSON.stringify({ name: 'build_squad', arguments: { budget: 1000 } }) + '\n';
          controller.enqueue(encoder.encode(toolCall));

          // Synchronously produce a mocked tool result payload
          const mockedResult = { squad: [], explanation: { mocked: true } };
          const dataFrame = 'd:' + JSON.stringify(mockedResult) + '\n';
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

// No need to mock data-service; the mocked AI stream does not execute the tool.

import { POST } from './route';

describe('/api/chat (tools mode) — mocked stream', () => {
  it('emits tool-call and tool-result frames with expected JSON shape', async () => {
    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'tools',
        messages: [
          { role: 'system', content: 'Test' },
          { role: 'user', content: 'Please build a squad with budget 1000.' },
        ],
      }),
    });

    const res = await POST(request);
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

    expect(output).toMatch(/(^|\n)f:\{/); // tool call frame exists
    expect(output).toMatch(/(^|\n)d:\{/); // tool result frame exists

    const dLine = output.split('\n').find((l) => l.startsWith('d:'));
    expect(dLine).toBeDefined();
    if (dLine) {
      const payload = JSON.parse(dLine.slice(2));
      expect(payload).toHaveProperty('squad');
      expect(payload).toHaveProperty('explanation');
      expect(payload.explanation).toHaveProperty('mocked', true);
    }
  });
});
