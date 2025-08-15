import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function listFixturesTool(api: FplApi) {
  return {
    description:
      "List FPL fixtures with optional filtering by gameweek, team, or completion status.",
    parameters: z.object({
      gameweek: z.number().int().positive().optional().describe("Filter by specific gameweek"),
      team: z.number().int().optional().describe("Filter by team id (shows fixtures involving this team)"),
      finished: z.boolean().optional().describe("Filter by completion status (true for finished, false for upcoming)"),
      limit: z.number().int().min(1).max(100).optional().describe("Max fixtures to return (1-100). Default 20"),
    }),
    execute: async ({ gameweek, team, finished, limit }: {
      gameweek?: number;
      team?: number;
      finished?: boolean;
      limit?: number;
    }) => {
      console.log('[list_fixtures] Starting execution with params:', { gameweek, team, finished, limit });
      
      try {
        console.log('[list_fixtures] Calling api.fixtures.fixturesList()...');
        const res = await api.fixtures.fixturesList();
        console.log('[list_fixtures] API response received:', {
          success: !!res,
          hasData: !!res?.data,
          dataType: typeof res?.data,
          dataLength: Array.isArray(res?.data) ? res.data.length : 'not array'
        });
        
        const fixtures = Array.isArray(res.data) ? res.data : [];
        console.log('[list_fixtures] Parsed fixtures count:', fixtures.length);

        // Apply filters
        const filtered = fixtures.filter((fixture) => {
          if (typeof gameweek === 'number' && fixture.gameweek !== gameweek) return false;
          if (typeof team === 'number' && fixture.teamHome !== team && fixture.teamAway !== team) return false;
          if (typeof finished === 'boolean' && fixture.finished !== finished) return false;
          return true;
        });

        console.log('[list_fixtures] After filtering:', filtered.length, 'fixtures');

        // Apply limit
        const limitValue = typeof limit === 'number' ? Math.min(Math.max(limit, 1), 100) : 20;
        const capped = filtered.slice(0, limitValue);
        console.log('[list_fixtures] After limit:', capped.length, 'fixtures');

        const result = capped.map((fixture) => ({
          id: fixture.id,
          gameweek: fixture.gameweek,
          season: fixture.season,
          teamHome: fixture.teamHome,
          teamAway: fixture.teamAway,
          teamHomeDifficulty: fixture.teamHomeDifficulty,
          teamAwayDifficulty: fixture.teamAwayDifficulty,
          kickoffTime: fixture.kickoffTime,
          finished: fixture.finished,
          teamHomeScore: fixture.teamHomeScore,
          teamAwayScore: fixture.teamAwayScore,
        }));
        
        console.log('[list_fixtures] Returning result with', result.length, 'fixtures');
        return result;
      } catch (error) {
        console.error('[list_fixtures] Error calling FPL API:', error);
        throw error;
      }
    },
  } as const;
}
