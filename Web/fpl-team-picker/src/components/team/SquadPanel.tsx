"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { dataService, type NormalizedMyTeam, type NormalizedPlayer } from "@/lib/data-service";
import { Position } from "@/models/position";
import { LoadingCard } from "@/components/utils/Loading";
import { Button } from "@/components/ui/button";

// Accept optional tool result via props
export type SquadPanelProps = {
  toolSquad?: any | null; // raw selectedSquad/squad from tool result
  header?: string;
};

function formatMoneyTenths(v: number | null | undefined) {
  if (v == null) return "-";
  return `£${(v / 10).toFixed(1)}`;
}

function budgetPct(totalCost: number, max = 1000) {
  const pct = Math.max(0, Math.min(100, (totalCost / max) * 100));
  return pct;
}

function positionLabel(p: Position) {
  switch (p) {
    case Position.GK: return "GK";
    case Position.DEF: return "DEF";
    case Position.MID: return "MID";
    case Position.FWD: return "FWD";
    default: return String(p);
  }
}

export default function SquadPanel({ toolSquad = null, header = "Wildcard Squad" }: SquadPanelProps) {
  const [data, setData] = useState<NormalizedMyTeam | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Record<Position, boolean>>({
    [Position.GK]: true,
    [Position.DEF]: true,
    [Position.MID]: true,
    [Position.FWD]: true,
  });

  // Local editable state: XI, bench, captain/vice
  const [xi, setXi] = useState<NormalizedPlayer[]>([]);
  const [bench, setBench] = useState<NormalizedPlayer[]>([]);
  const [captainId, setCaptainId] = useState<number | null>(null);
  const [viceId, setViceId] = useState<number | null>(null);
  // Player details modal state
  const [selected, setSelected] = useState<NormalizedPlayer | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let res: NormalizedMyTeam;
      if (toolSquad) {
        res = await dataService.fromToolSquad(toolSquad);
      } else {
        res = await dataService.getWildcardRecommendation();
      }
      setData(res);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load recommendation");
    } finally {
      setLoading(false);
    }
  }, [toolSquad]);

  useEffect(() => { refresh(); }, [refresh]);

  // When normalized data arrives, seed local editable state
  useEffect(() => {
    const squad = data?.squad;
    if (!squad) return;
    setXi(squad.startingXi || []);
    setBench(squad.bench || []);
    setCaptainId(squad.captain?.id ?? (squad.startingXi?.[0]?.id ?? null));
    // default vice: first non-captain if present
    const firstNonCaptain = (squad.startingXi || []).find(p => p.id !== (squad.captain?.id ?? -1));
    setViceId(squad.viceCaptain?.id ?? (firstNonCaptain?.id ?? null));
  }, [data]);

  const allPlayers: NormalizedPlayer[] = useMemo(() => {
    return [...xi, ...bench];
  }, [xi, bench]);

  const filteredXi = useMemo(() => {
    return xi.filter(p => filters[p.position as Position]);
  }, [xi, filters]);

  const filteredBench = useMemo(() => {
    return bench.filter(p => filters[p.position as Position]);
  }, [bench, filters]);

  const clubCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of allPlayers) {
      const key = p.team.shortName;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [allPlayers]);

  const sumXp = (arr: NormalizedPlayer[]) => arr.reduce((acc, p) => acc + (p.expectedPoints ?? 0), 0);
  const captain = useMemo(() => xi.find(p => p.id === captainId) || null, [xi, captainId]);
  const predictedPoints = useMemo(() => {
    const base = sumXp(xi);
    const capBonus = captain ? (captain.expectedPoints ?? 0) : 0; // double captain
    return base + capBonus;
  }, [xi, captain]);
  const benchBoostPredictedPoints = useMemo(() => {
    return predictedPoints + sumXp(bench);
  }, [predictedPoints, bench]);

  const makeCaptain = (id: number) => {
    if (id === viceId) setViceId(captainId); // swap to keep distinct
    setCaptainId(id);
  };
  const makeVice = (id: number) => {
    if (id === captainId) setCaptainId(viceId); // swap to keep distinct
    setViceId(id);
  };

  const moveBench = (index: number, dir: -1 | 1) => {
    setBench(prev => {
      const next = prev.slice();
      const newIndex = index + dir;
      if (newIndex < 0 || newIndex >= next.length) return prev;
      const tmp = next[index];
      next[index] = next[newIndex];
      next[newIndex] = tmp;
      return next;
    });
  };

  if (loading) return (
    <div className="h-[70vh] border rounded-md p-3 flex flex-col gap-3 bg-card animate-pulse" aria-busy="true" aria-live="polite">
      <div className="flex items-center justify-between">
        <div>
          <div className="h-5 w-40 rounded bg-muted" />
          <div className="mt-1 h-3 w-52 rounded bg-muted" />
        </div>
        <div className="h-8 w-20 rounded bg-muted" />
      </div>

      <div>
        <div className="mb-1 h-3 w-24 rounded bg-muted" />
        <div className="h-2 w-full rounded bg-muted" />
      </div>

      <div className="flex flex-wrap gap-2">
        <div className="h-6 w-10 rounded bg-muted" />
        <div className="h-6 w-12 rounded bg-muted" />
        <div className="h-6 w-12 rounded bg-muted" />
        <div className="h-6 w-10 rounded bg-muted" />
      </div>

      <div className="flex-1 min-h-0 overflow-hidden rounded-md border bg-background/50">
        <div className="divide-y">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="flex items-center justify-between px-2 py-2">
              <div className="h-4 w-40 rounded bg-muted" />
              <div className="h-4 w-16 rounded bg-muted" />
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-end">
        <div className="h-4 w-40 rounded bg-muted" />
      </div>
    </div>
  );
  if (error) return (
    <div className="h-[70vh] border rounded-md p-4 flex flex-col gap-2">
      <div className="text-sm text-destructive">{error}</div>
      <div>
        <Button size="sm" onClick={refresh} disabled={loading}>Retry</Button>
      </div>
    </div>
  );

  return (
    <div className="h-[70vh] border rounded-md p-3 flex flex-col gap-3 bg-card">
      {/* Details modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => setSelected(null)} />
          <div className="relative z-10 w-[92vw] max-w-md rounded-md border bg-background p-4 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-lg font-semibold">{selected.name}</div>
                <div className="text-xs text-muted-foreground">{positionLabel(selected.position)} • {selected.team.shortName} • {formatMoneyTenths(selected.cost)}</div>
              </div>
              <Button size="sm" variant="outline" onClick={() => setSelected(null)}>Close</Button>
            </div>
            <div className="grid gap-2 text-sm">
              <div className="flex justify-between"><span>Expected points (GW):</span><span className="font-mono">{selected.expectedPoints?.toFixed(1) ?? '-'}</span></div>
              <div className="flex justify-between"><span>Availability:</span><span>{selected.availability.isAvailable ? 'Available' : 'Doubtful'} ({selected.availability.chanceOfPlaying ?? '-'}%)</span></div>
              <div className="flex justify-between"><span>Selected by %:</span><span className="font-mono">{(selected.stats.selectedByPercent ?? 0).toFixed(1)}%</span></div>
              <div className="flex justify-between"><span>Season points:</span><span className="font-mono">{selected.stats.seasonPoints}</span></div>
            </div>
            {selected.predictions && Object.keys(selected.predictions).length > 0 && (
              <div className="mt-3">
                <div className="text-xs font-medium text-muted-foreground mb-1">Upcoming predictions</div>
                <div className="rounded-md border divide-y">
                  {Object.entries(selected.predictions).slice(0, 6).map(([gw, val]) => (
                    <div key={gw} className="flex justify-between px-2 py-1 text-sm">
                      <span>GW {gw}</span>
                      <span className="font-mono">{Number(val).toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{header}</h2>
          <p className="text-xs text-muted-foreground">{toolSquad ? 'From chat tool result' : 'Live recommendation from optimizer'}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={refresh} disabled={loading}>Refresh</Button>
        </div>
      </div>

      {/* Budget bar */}
      <div>
        <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
          <span>Budget used</span>
          <span>{formatMoneyTenths((data?.squad?.totalCost) ?? null)} / £100.0</span>
        </div>
        <div className="h-2 w-full rounded-md bg-muted overflow-hidden">
          <div
            className="h-full bg-blue-500"
            style={{ width: `${budgetPct((data?.squad?.totalCost) || 0)}%` }}
          />
        </div>
      </div>

      {/* Position filters */}
      <div className="flex flex-wrap gap-2">
        {(Object.values(Position).filter(v => typeof v === "number") as number[]).map((p) => (
          <button
            key={p}
            onClick={() => setFilters(f => ({ ...f, [p]: !f[p as Position] }))}
            className={`px-2 py-1 rounded-md text-xs border ${filters[p as Position] ? "bg-primary text-primary-foreground" : "bg-background text-foreground"}`}
          >
            {positionLabel(p as Position)}
          </button>
        ))}
      </div>

      {/* Club caps summary */}
      <div className="flex flex-wrap gap-1 text-xs">
        {clubCounts.map(([club, count]) => (
          <span key={club} className={`px-2 py-0.5 rounded-full border ${count >= 3 ? "bg-amber-100 text-amber-900 dark:bg-amber-900/30 dark:text-amber-200" : "text-muted-foreground"}`}>
            {club}: {count}
          </span>
        ))}
      </div>

      {/* XI */}
      <div className="flex-1 min-h-0 overflow-auto rounded-md border bg-background/50">
        <table className="w-full text-sm">
          <tbody>
            <tr className="font-mono text-xs text-muted-foreground uppercase text-center"><td className="p-2" colSpan={99}>XI</td></tr>
            {filteredXi.map((p, idx) => (
              <tr key={`xi-${idx}`} className="hover:bg-muted/40">
                <td className="px-2 py-1 font-medium flex items-center gap-2">
                  {p.name}
                  {p.id === captainId && <span className="text-xs px-1 rounded bg-blue-600 text-white">C</span>}
                  {p.id === viceId && <span className="text-xs px-1 rounded bg-slate-500 text-white">VC</span>}
                </td>
                <td className="px-2 py-1 text-xs text-muted-foreground">{positionLabel(p.position)}</td>
                <td className="px-2 py-1 text-xs">{p.team.shortName}</td>
                <td className="px-2 py-1 text-right font-mono text-xs text-gray-500">{formatMoneyTenths(p.cost)}</td>
                <td className="px-2 py-1 text-right font-mono text-xs text-blue-500">{p.expectedPoints?.toFixed(1) ?? "-"}</td>
                <td className="px-2 py-1 text-right">
                  <div className="flex items-center justify-end gap-2">
                    <Button size="sm" variant={p.id === captainId ? "default" : "outline"} onClick={() => makeCaptain(p.id)}>Set C</Button>
                    <Button size="sm" variant={p.id === viceId ? "default" : "outline"} onClick={() => makeVice(p.id)}>Set VC</Button>
                    <Button size="sm" variant="ghost" onClick={() => setSelected(p)}>Details</Button>
                  </div>
                </td>
              </tr>
            ))}
            {filteredXi.length === 0 && (
              <tr>
                <td className="p-3 text-center text-xs text-muted-foreground" colSpan={99}>
                  No XI players match current filters.
                </td>
              </tr>
            )}
            <tr className="font-mono text-xs text-muted-foreground uppercase text-center"><td className="p-2" colSpan={99}>Bench</td></tr>
            {filteredBench.map((p, idx) => (
              <tr key={`bench-${idx}`} className="hover:bg-muted/40">
                <td className="px-2 py-1 font-medium">{p.name}</td>
                <td className="px-2 py-1 text-xs text-muted-foreground">{positionLabel(p.position)}</td>
                <td className="px-2 py-1 text-xs">{p.team.shortName}</td>
                <td className="px-2 py-1 text-right font-mono text-xs text-gray-500">{formatMoneyTenths(p.cost)}</td>
                <td className="px-2 py-1 text-right font-mono text-xs text-blue-500">{p.expectedPoints?.toFixed(1) ?? "-"}</td>
                <td className="px-2 py-1 text-right">
                  <div className="flex items-center justify-end gap-1">
                    <Button size="sm" variant="outline" disabled={idx === 0} onClick={() => moveBench(idx, -1)}>↑</Button>
                    <Button size="sm" variant="outline" disabled={idx === filteredBench.length - 1} onClick={() => moveBench(idx, 1)}>↓</Button>
                    <Button size="sm" variant="ghost" onClick={() => setSelected(p)}>Details</Button>
                  </div>
                </td>
              </tr>
            ))}
            {filteredBench.length === 0 && (
              <tr>
                <td className="p-3 text-center text-xs text-muted-foreground" colSpan={99}>
                  No bench players match current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Totals */}
      <div className="flex justify-end text-blue-500 text-sm">
        <div className="mr-1">XP (bench boost):</div>
        <div className="font-mono">
          {predictedPoints.toFixed(1)} ({benchBoostPredictedPoints.toFixed(1)})
        </div>
      </div>
    </div>
  );
}
