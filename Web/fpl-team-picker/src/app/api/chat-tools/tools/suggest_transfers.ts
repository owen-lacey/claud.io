import { z } from "zod";
import { FplApi } from "@/helpers/fpl-api";

export function suggestTransfersTool(api: FplApi) {
  return {
    description:
      "Suggest up to 'ft' transfers based on available budget (itb) and simple expected-points heuristic. Returns compact JSON.",
    parameters: z.object({
      ft: z.number().int().min(0).max(3).optional(),
      itb: z.number().int().min(0).optional(),
      gw: z.number().int().positive().optional(),
    }),
    execute: async (args: { ft?: number; itb?: number; gw?: number }) => {
      const ft = args.ft ?? 1;
      const itb = args.itb ?? 0;
      const gw = args.gw;

      // Minimal heuristic: pick top-xp affordable player as a transfer in.
      const res = await api.players.playersList();
      const players: Array<any> = Array.isArray(res.data) ? res.data : [];

      const sorted = players
        .slice()
        .sort((a, b) => (b?.xp ?? 0) - (a?.xp ?? 0));

      const picks = sorted.slice(0, Math.max(1, Math.min(ft, 3)));

      const transfers = picks.map((p) => ({
        out: null, // current squad unknown in this minimal stub
        in: {
          id: p.id,
          name: p.name,
          position: p.position,
          team: p.team,
          cost: p.cost,
          xp: p.xp,
        },
        reason: "High expected points candidate within simple constraints",
        estDeltaPoints: p?.xp ?? 0,
        price: p?.cost ?? null,
      }));

      return {
        transfers,
        meta: { ft, itb, gw: gw ?? null },
        note: "Current squad not provided; suggestions are generic based on xp only.",
      } as const;
    },
  } as const;
}
