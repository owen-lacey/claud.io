"use client";

import { useEffect, useMemo, useState } from "react";
import { dataService, type NormalizedMyTeam, type NormalizedPlayer, type SelectionContext } from "@/lib/data-service";
import { Position } from "@/models/position";
import { Button } from "@/components/ui/button";
import { LoadingCard } from "@/components/utils/Loading";

type Transfer = {
  out: NormalizedPlayer;
  in: NormalizedPlayer;
  deltaPoints: number; // in expected points
  netSpend: number; // in tenths (positive means spend, negative means saving)
  note: string;
};

function formatMoneyTenths(v: number) {
  return `£${(v / 10).toFixed(1)}`;
}

function posLabel(p: Position) {
  switch (p) {
    case Position.GK: return "GK";
    case Position.DEF: return "DEF";
    case Position.MID: return "MID";
    case Position.FWD: return "FWD";
    default: return String(p);
  }
}

function computeSuggestions(ctx: SelectionContext): Transfer[] {
  const my = ctx.myTeam;
  const squad = my.squad;
  if (!squad) return [];

  let ft = Math.max(0, Math.min(3, my.freeTransfers ?? 1));
  let itb = my.bank ?? 0;

  const inSquad = new Set<number>([...squad.startingXi, ...squad.bench].map(p => p.id));
  const clubCounts = new Map<string, number>();
  for (const p of [...squad.startingXi, ...squad.bench]) {
    clubCounts.set(p.team.shortName, (clubCounts.get(p.team.shortName) || 0) + 1);
  }

  const sortedCandidates = ctx.players
    .filter(p => p.expectedPoints != null && p.availability.isAvailable && !inSquad.has(p.id))
    .sort((a, b) => (b.expectedPoints! - a.expectedPoints!));

  const byPosOut: Record<Position, NormalizedPlayer[]> = {
    [Position.GK]: [],
    [Position.DEF]: [],
    [Position.MID]: [],
    [Position.FWD]: [],
  };
  for (const p of [...squad.startingXi, ...squad.bench]) {
    byPosOut[p.position as Position].push(p);
  }
  for (const k of Object.keys(byPosOut)) {
    const pos = Number(k) as Position;
    byPosOut[pos] = byPosOut[pos].sort((a, b) => (a.expectedPoints ?? 0) - (b.expectedPoints ?? 0));
  }

  const picks: Transfer[] = [];

  for (const cand of sortedCandidates) {
    if (ft <= 0) break;
    const pos = cand.position as Position;

    const outs = byPosOut[pos];
    let chosenOut: NormalizedPlayer | undefined;

    for (const out of outs) {
      // Club cap check: at most 3 per club for candidate's club after the swap
      const candClub = cand.team.shortName;
      const currentCandClubCount = clubCounts.get(candClub) || 0;
      const afterCandClubCount = currentCandClubCount + (out.team.shortName === candClub ? 0 : 1);
      if (afterCandClubCount > 3) continue;

      // Budget check
      const netSpend = cand.cost - out.cost;
      if (netSpend > itb) continue;

      chosenOut = out;
      break;
    }

    if (!chosenOut) continue;

    // Commit transfer
    const netSpend = cand.cost - chosenOut.cost;
    const delta = (cand.expectedPoints ?? 0) - (chosenOut.expectedPoints ?? 0);
    picks.push({ out: chosenOut, in: cand, deltaPoints: delta, netSpend, note: "xp upgrade within budget and club cap" });

    // Update state
    ft -= 1;
    itb -= Math.max(0, netSpend);

    // Update club counts
    const outClub = chosenOut.team.shortName;
    const inClub = cand.team.shortName;
    clubCounts.set(outClub, (clubCounts.get(outClub) || 0) - 1);
    clubCounts.set(inClub, (clubCounts.get(inClub) || 0) + 1);

    // Remove used players from pools
    byPosOut[pos] = outs.filter(p => p.id !== chosenOut!.id);
  }

  return picks.slice(0, 3);
}

export default function TransferSuggester() {
  const [ctx, setCtx] = useState<SelectionContext | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [transfers, setTransfers] = useState<Transfer[]>([]);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const c = await dataService.getSelectionContext();
      setCtx(c);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load context");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    if (!ctx) return;
    const recs = computeSuggestions(ctx);
    setTransfers(recs);
  }, [ctx]);

  if (loading) return (
    <div className="border rounded-md p-3 flex flex-col gap-3 bg-card animate-pulse" aria-busy="true" aria-live="polite">
      <div className="flex items-center justify-between">
        <div>
          <div className="h-5 w-40 rounded bg-muted" />
          <div className="mt-1 h-3 w-52 rounded bg-muted" />
        </div>
        <div className="h-8 w-20 rounded bg-muted" />
      </div>
      <div className="flex flex-col gap-2">
        {Array.from({ length: 2 }).map((_, i) => (
          <div key={i} className="rounded-md border bg-background/50 p-2">
            <div className="flex items-center justify-between">
              <div className="h-4 w-32 rounded bg-muted" />
              <div className="h-4 w-20 rounded bg-muted" />
            </div>
            <div className="mt-2 h-3 w-40 rounded bg-muted" />
          </div>
        ))}
      </div>
    </div>
  );
  if (error) return (
    <div className="border rounded-md p-4 flex flex-col gap-2">
      <div className="text-sm text-destructive">{error}</div>
      <div>
        <Button size="sm" onClick={refresh} disabled={loading}>Retry</Button>
      </div>
    </div>
  );

  const ft = ctx?.myTeam.freeTransfers ?? 1;
  const itb = ctx?.myTeam.bank ?? 0;

  return (
    <div className="border rounded-md p-3 flex flex-col gap-3 bg-card">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Transfer Suggestions</h2>
          <p className="text-xs text-muted-foreground">Up to {ft} FT, ITB {formatMoneyTenths(itb)}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={refresh} disabled={loading}>Refresh</Button>
        </div>
      </div>

      {transfers.length === 0 ? (
        <div className="text-sm text-muted-foreground">No clear upgrades found within constraints.</div>
      ) : (
        <div className="flex flex-col gap-2">
          {transfers.map((t, idx) => (
            <div key={idx} className="rounded-md border bg-background/50 p-2 text-sm">
              <div className="flex items-center justify-between">
                <div className="font-medium">
                  {t.out.name} → {t.in.name}
                </div>
                <div className="font-mono">
                  +{t.deltaPoints.toFixed(1)} xp · {t.netSpend >= 0 ? "+" : ""}{formatMoneyTenths(t.netSpend)}
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                {posLabel(t.out.position)} · {t.out.team.shortName} → {t.in.team.shortName} · {t.note}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
