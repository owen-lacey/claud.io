import { z } from "zod";

export function explainSelectionTool() {
  return {
    description:
      "Explain a given squad or a set of transfers. Provide concise rationale based on expected points and constraints.",
    parameters: z.object({
      squad: z
        .object({
          startingXi: z.array(
            z.object({ id: z.number(), name: z.string().optional(), position: z.number(), team: z.any().optional(), cost: z.number().optional(), expectedPoints: z.number().optional() })
          ),
          bench: z.array(
            z.object({ id: z.number(), name: z.string().optional(), position: z.number(), team: z.any().optional(), cost: z.number().optional(), expectedPoints: z.number().optional() })
          ),
          totalCost: z.number().optional(),
          predictedPoints: z.number().optional(),
        })
        .optional(),
      transfers: z
        .array(
          z.object({
            out: z.object({ id: z.number(), name: z.string().optional() }).nullable().optional(),
            in: z.object({ id: z.number(), name: z.string().optional(), position: z.number().optional(), cost: z.number().optional(), xp: z.number().optional() }),
          })
        )
        .optional(),
    }),
    execute: async (args: any) => {
      const notes: string[] = [];
      const picks: Array<{ id: number; reason: string }> = [];

      if (args?.squad) {
        const starters = args.squad.startingXi || [];
        starters.forEach((p: any, idx: number) => {
          const why = idx === 0 ? "Captained due to top projected points and reliability" : "Selected for strong projection and role";
          picks.push({ id: p.id, reason: why });
        });
        notes.push(
          `Predicted points: ${args.squad.predictedPoints ?? starters.reduce((s: number, p: any) => s + (p.expectedPoints ?? 0), 0)}`
        );
      }

      if (Array.isArray(args?.transfers) && args.transfers.length > 0) {
        args.transfers.forEach((t: any) => {
          const name = t?.in?.name ?? `#${t?.in?.id}`;
          picks.push({ id: t?.in?.id, reason: `${name} improves expected points` });
        });
        notes.push("Transfers aim to increase projected points while keeping budget constraints in mind.");
      }

      if (notes.length === 0 && picks.length === 0) {
        notes.push("No inputs provided; nothing to explain.");
      }

      return { notes, picks } as const;
    },
  } as const;
}
