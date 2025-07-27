import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai('gpt-4o-mini'),
    messages,
    system: `You are a helpful Fantasy Premier League (FPL) assistant. You can help users with:
    - Player recommendations and analysis
    - Team selection strategies
    - Transfer advice
    - League management tips
    - Current gameweek insights
    - Price changes and fixture analysis
    
    Be helpful, knowledgeable, and provide specific FPL advice when possible.`,
  });

  return result.toDataStreamResponse();
}
