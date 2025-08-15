import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function listTeamsTool(api: FplApi) {
  return {
    description:
      "List all FPL teams with their basic information (id, name, short name, code).",
    parameters: z.object({
      // No parameters needed - just returns all teams
    }),
    execute: async () => {
      console.log('[list_teams] Starting execution...');
      
      try {
        console.log('[list_teams] Calling api.teams.teamsList()...');
        const res = await api.teams.teamsList();
        console.log('[list_teams] API response received:', {
          success: !!res,
          hasData: !!res?.data,
          dataType: typeof res?.data,
          dataLength: Array.isArray(res?.data) ? res.data.length : 'not array'
        });
        
        const teams = Array.isArray(res.data) ? res.data : [];
        console.log('[list_teams] Parsed teams count:', teams.length);

        const result = teams.map((team) => ({
          id: team.id,
          name: team.name,
          shortName: team.shortName,
          code: team.code,
        }));
        
        console.log('[list_teams] Returning result with', result.length, 'teams');
        return result;
      } catch (error) {
        console.error('[list_teams] Error calling FPL API:', error);
        throw error;
      }
    },
  } as const;
}
