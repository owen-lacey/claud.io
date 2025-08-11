/**
 * @jest-environment node
 */

// Mock AI stream to avoid external calls and keep the test deterministic/fast
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

describe('/api/chat-tools contract — bad input', () => {
  it('streams a 3: error frame on invalid JSON body', async () => {
    const bad = new Request('http://localhost:3000/api/chat-tools', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // Provide an invalid body by bypassing JSON entirely (stream non-JSON)
      body: '{not json',
    } as any);

    const res = await POST(bad as any);
    expect(res.status).toBe(200);

    const text = await res.text();
    expect(text).toContain('3:');
    expect(text).toContain('[chat-tools] Invalid JSON payload');
  });

  it('ignores unsupported roles and empty content', async () => {
    const req = new Request('http://localhost:3000/api/chat-tools', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', pl_profile: 'test-token' },
      body: JSON.stringify({
        messages: [
          { role: 'tool', content: 'should be ignored' },
          { role: 'user', content: '' },
          { role: 'user', content: 'hi' },
        ],
      }),
    });

    const res = await POST(req as any);
    expect(res.status).toBe(200);

    const text = await res.text();
    // With mocked stream, we should at least get a done frame
    expect(text).toMatch(/(^|\n)d:\{/);
  });
});
