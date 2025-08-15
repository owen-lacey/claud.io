import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function getMyOverviewTool(api: FplApi, token?: string) {
  return {
    description: 'Return a summary of the current user and their leagues (counts and basic league info).',
    parameters: z.object({}),
    execute: async () => {
      console.log('[get_my_overview] Starting execution, token present:', !!token);
      
      if (!token) {
        console.error('[get_my_overview] No token provided');
        throw new Error('Missing authentication: expected pl_profile or Authorization header');
      }
      
      try {
        console.log('[get_my_overview] Making API calls to myDetails and myLeagues...');
        const [userRes, leaguesRes] = await Promise.all([
          api.myDetails.myDetailsList(),
          api.myLeagues.myLeaguesList(),
        ]);
        
        console.log('[get_my_overview] API responses received:', {
          userSuccess: !!userRes,
          userHasData: !!userRes?.data,
          leaguesSuccess: !!leaguesRes,
          leaguesHasData: !!leaguesRes?.data,
          leaguesIsArray: Array.isArray(leaguesRes?.data),
          leaguesLength: Array.isArray(leaguesRes?.data) ? leaguesRes.data.length : 'not array'
        });
        
        const user = userRes.data || {};
        const leagues = Array.isArray(leaguesRes.data) ? leaguesRes.data : [];
        const items = leagues.map((l) => ({
          id: l.id ?? undefined,
          name: l.name ?? undefined,
          currentPosition: l.currentPosition ?? undefined,
          numberOfPlayers: l.numberOfPlayers ?? undefined,
        }));
        
        const result = {
          user: {
            id: user.id ?? undefined,
            firstName: user.firstName ?? undefined,
            lastName: user.lastName ?? undefined,
          },
          leagues: {
            count: items.length,
            items,
          },
        };
        
        console.log('[get_my_overview] Returning result:', {
          userHasId: !!result.user.id,
          leaguesCount: result.leagues.count
        });
        
        return result;
      } catch (err: any) {
        console.error('[get_my_overview] Error occurred:', err);
        const status = err?.response?.status;
        const msg = err?.message || 'Backend call failed';
        const errorMsg = `[get_my_overview] ${msg}${status ? ` (status=${status})` : ''}`;
        console.error('[get_my_overview] Throwing error:', errorMsg);
        throw new Error(errorMsg);
      }
    }
  } as const;
}
