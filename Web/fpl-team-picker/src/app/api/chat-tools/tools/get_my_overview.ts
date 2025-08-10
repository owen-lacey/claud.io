import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function getMyOverviewTool(api: FplApi, token?: string) {
  return {
    description: 'Return a summary of the current user and their leagues (counts and basic league info).',
    parameters: z.object({}),
    execute: async () => {
      if (!token) {
        throw new Error('Missing authentication: expected pl_profile or Authorization header');
      }
      try {
        const [userRes, leaguesRes] = await Promise.all([
          api.myDetails.myDetailsList(),
          api.myLeagues.myLeaguesList(),
        ]);
        const user = userRes.data || {};
        const leagues = Array.isArray(leaguesRes.data) ? leaguesRes.data : [];
        const items = leagues.map((l) => ({
          id: l.id ?? undefined,
          name: l.name ?? undefined,
          currentPosition: l.currentPosition ?? undefined,
          numberOfPlayers: l.numberOfPlayers ?? undefined,
        }));
        return {
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
      } catch (err: any) {
        const status = err?.response?.status;
        const msg = err?.message || 'Backend call failed';
        throw new Error(`[get_my_overview] ${msg}${status ? ` (status=${status})` : ''}`);
      }
    }
  } as const;
}
