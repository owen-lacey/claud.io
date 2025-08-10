import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function buildSquadTool(api: FplApi, token?: string) {
  return {
    description: "Build an optimal FPL squad using the backend optimizer (wildcard). Returns the selected squad.",
    parameters: z.object({}),
    execute: async () => {
      // Basic auth check, keep error helpful for UI
      if (!token) {
        throw new Error('Missing authentication: expected pl_profile or Authorization header');
      }
      try {
        console.log('[chat-tools] build_squad: calling backend /wildcard');
        const res = await api.wildcard.wildcardCreate();
        const squad = res.data?.selectedSquad ?? null;
        if (!squad) {
          throw new Error('No squad returned from backend');
        }
        return squad;
      } catch (err: any) {
        const status = err?.response?.status;
        const msg = err?.message || 'Backend call failed';
        throw new Error(`[build_squad] ${msg}${status ? ` (status=${status})` : ''}`);
      }
    },
  } as const;
}
