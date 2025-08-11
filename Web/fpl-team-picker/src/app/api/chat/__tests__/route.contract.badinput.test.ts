/**
 * @jest-environment node
 */

// Mock AI stream so the test is deterministic and fast
jest.mock('ai', () => {
  return {
    streamText: () => {
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          const doneFrame = 'd:' + JSON.stringify({ finishReason: 'stop', usage: { promptTokens: 1, completionTokens: 0 } }) + '\n';
          controller.enqueue(encoder.encode(doneFrame));
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

import { POST } from '../route';

describe('/api/chat contract — bad input', () => {
  it('streams a 3: error frame on invalid JSON body', async () => {
    const bad = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{not json',
    } as any);

    const res = await POST(bad as any);
    expect(res.status).toBe(200);

    const text = await res.text();
    expect(text).toContain('3:');
    expect(text).toContain('[chat] Invalid JSON payload');
  });

  it('sanitizes roles and content before sending to the model', async () => {
    const req = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [
          { role: 'tool', content: 'ignore' },
          { role: 'user', content: '' },
          { role: 'assistant', content: [{ text: 'hello' }] },
          { role: 'user', content: 'world' },
        ],
      }),
    });

    const res = await POST(req as any);
    expect(res.status).toBe(200);

    const text = await res.text();
    expect(text).toMatch(/(^|\n)d:\{/);
  });
});
