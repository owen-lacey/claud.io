import { google } from "@ai-sdk/google";
import { streamText } from "ai";
import { z } from "zod";

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  // Minimal generic tool, force tool invocation
  const result = streamText({
    model: google('gemini-2.0-flash'),
    messages,
    tools: {
      queryDatabase: {
        description: "Return a random integer in the inclusive range [min, max]. Defaults to 1-100.",
        parameters: z.object({
          min: z.number().int().optional().describe('Minimum integer (inclusive). Default 1.'),
          max: z.number().int().optional().describe('Maximum integer (inclusive). Default 100.'),
        }),
        execute: async ({ min, max }: { min?: number; max?: number }) => {
          console.log('[chat-tools] Simulating tool call', { min, max });
          // Determine effective bounds
          let a = Number.isFinite(min as number) ? Math.trunc(min as number) : 1;
          let b = Number.isFinite(max as number) ? Math.trunc(max as number) : 100;
          if (Number.isNaN(a)) a = 1;
          if (Number.isNaN(b)) b = 100;
          if (a > b) [a, b] = [b, a];
          // Generate random integer in [a, b]
          const value = Math.floor(Math.random() * (b - a + 1)) + a;
          return { result: value, range: { min: a, max: b } };
        }
      }
    }
  });

  return result.toDataStreamResponse();
}
