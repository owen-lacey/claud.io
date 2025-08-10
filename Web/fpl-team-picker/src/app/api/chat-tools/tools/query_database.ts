import { z } from "zod";

export function queryDatabaseTool() {
  return {
    description: "Return a random integer in the inclusive range [min, max]. Defaults to 1-100.",
    parameters: z.object({
      min: z.number().int().optional().describe('Minimum integer (inclusive). Default 1.'),
      max: z.number().int().optional().describe('Maximum integer (inclusive). Default 100.'),
    }),
    execute: async ({ min, max }: { min?: number; max?: number }) => {
      console.log('[chat-tools] Simulating tool call', { min, max });
      let a = Number.isFinite(min as number) ? Math.trunc(min as number) : 1;
      let b = Number.isFinite(max as number) ? Math.trunc(max as number) : 100;
      if (Number.isNaN(a)) a = 1;
      if (Number.isNaN(b)) b = 100;
      if (a > b) [a, b] = [b, a];
      const value = Math.floor(Math.random() * (b - a + 1)) + a;
      return { result: value, range: { min: a, max: b } };
    }
  } as const;
}
