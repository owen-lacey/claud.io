import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function listPlayersTool(api: FplApi) {
  return {
    description:
      "List players matching simple criteria. Returns compact fields only (id, name, position, team, cost, xp, selectedByPercent).",
    parameters: z.object({
      position: z.union([z.number().int(), z.string()]).optional().describe("Position id (1..5) or name: GK, DEF, MID, FWD"),
      team: z.number().int().optional().describe("Team id"),
      minXp: z.number().optional().describe("Minimum projected points/xp"),
      maxCost: z.number().optional().describe("Maximum price/cost"),
      limit: z.number().int().min(1).max(50).optional().describe("Max items (1..50). Default 10"),
    }),
    execute: async ({ position, team, minXp, maxCost, limit }: {
      position?: number | string;
      team?: number;
      minXp?: number;
      maxCost?: number;
      limit?: number;
    }) => {
      const res = await api.players.playersList();
      const players = Array.isArray(res.data) ? res.data : [];

      // normalize requested position to numeric id
      const posMap: Record<string, number> = { GK: 1, GKP: 1, DEF: 2, D: 2, MID: 3, M: 3, FWD: 4, F: 4 };
      let posId: number | undefined;
      if (typeof position === 'number') posId = position;
      else if (typeof position === 'string') posId = posMap[position.toUpperCase()];

      const filtered = players.filter((p) => {
        if (posId && p.position !== posId) return false;
        if (typeof team === 'number' && p.team !== team) return false;
        if (typeof minXp === 'number' && typeof p.xp === 'number' && p.xp < minXp) return false;
        if (typeof maxCost === 'number' && typeof p.cost === 'number' && p.cost > maxCost) return false;
        return true;
      });

      const capped = filtered.slice(0, typeof limit === 'number' ? Math.min(Math.max(limit, 1), 50) : 10);

      return capped.map((p) => ({
        id: p.id,
        name: p.name ?? `${p.firstName ?? ''} ${p.secondName ?? ''}`.trim(),
        position: p.position,
        team: p.team,
        cost: p.cost,
        xp: p.xp ?? undefined,
        selectedByPercent: p.selectedByPercent ?? undefined,
      }));
    },
  } as const;
}
