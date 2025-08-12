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
        <div className="h-[70vh] border border-border/50 rounded-xl p-4 flex flex-col gap-4 bg-card animate-pulse" aria-busy="true" aria-live="polite">
      <div className="flex items-center justify-between">
        <div>
          <div className="h-5 w-40 rounded-lg bg-muted/50" />
          <div className="mt-1 h-3 w-52 rounded-lg bg-muted/30" />
        </div>
        <div className="h-8 w-20 rounded-lg bg-muted/50" />
      </div>

      {/* Budget loading placeholder */}
      <div>
        <div className="mb-1 h-3 w-24 rounded-lg bg-muted/50" />
        <div className="h-2 w-full rounded-lg bg-muted/30" />
      </div>

      {/* Filter tabs loading placeholder */}
      <div className="flex flex-wrap gap-2">
        <div className="h-6 w-10 rounded-lg bg-muted/50" />
        <div className="h-6 w-12 rounded-lg bg-muted/50" />
        <div className="h-6 w-12 rounded-lg bg-muted/50" />
        <div className="h-6 w-10 rounded-lg bg-muted/50" />
      </div>

      {/* Players list loading placeholder */}
      <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-border/50 bg-background/50">
        <div className="divide-y divide-border/30">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="flex items-center justify-between px-3 py-3">
              <div className="h-4 w-40 rounded-lg bg-muted/50" />
              <div className="h-4 w-16 rounded-lg bg-muted/50" />
            </div>
          ))}
        </div>
      </div>

      {/* Team stats loading placeholder */}
      <div className="flex justify-end">
        <div className="h-4 w-40 rounded-lg bg-muted/50" />
      </div>
    </div>
  );
  if (error) return (
    <div className="h-[70vh] border border-border/50 rounded-xl p-4 flex flex-col gap-3 bg-card">
      <div className="text-sm text-destructive">{error}</div>
      <div>
        <Button size="sm" onClick={refresh} disabled={loading}>Retry</Button>
      </div>
    </div>
  );

  return (
    <div className="h-[70vh] border border-border/50 rounded-xl p-4 flex flex-col gap-4 bg-card">
      {/* Details modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/20 backdrop-blur-sm" onClick={() => setSelected(null)} />
          <div className="relative z-10 w-[92vw] max-w-md rounded-xl border border-border/50 bg-card p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-lg font-semibold text-foreground">{selected.name}</div>
                <div className="text-xs text-muted-foreground">{positionLabel(selected.position)} • {selected.team.shortName} • {formatMoneyTenths(selected.cost)}</div>
              </div>
              <Button size="sm" variant="outline" onClick={() => setSelected(null)}>Close</Button>
            </div>
            <div className="grid gap-3 text-sm">
              <div className="flex justify-between"><span className="text-muted-foreground">Expected points (GW):</span><span className="font-mono text-foreground">{selected.expectedPoints?.toFixed(1) ?? '-'}</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Availability:</span><span className="text-foreground">{selected.availability.isAvailable ? 'Available' : 'Doubtful'} ({selected.availability.chanceOfPlaying ?? '-'}%)</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Selected by %:</span><span className="font-mono text-foreground">{(selected.stats.selectedByPercent ?? 0).toFixed(1)}%</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Season points:</span><span className="font-mono text-foreground">{selected.stats.seasonPoints}</span></div>
            </div>
            {selected.predictions && Object.keys(selected.predictions).length > 0 && (
              <div className="mt-4">
                <div className="text-xs font-medium text-muted-foreground mb-2">Upcoming predictions</div>
                <div className="rounded-lg border border-border/50 divide-y divide-border/30 bg-muted/20">
                  {Object.entries(selected.predictions).slice(0, 6).map(([gw, val]) => (
                    <div key={gw} className="flex justify-between px-3 py-2 text-sm">
                      <span className="text-muted-foreground">GW {gw}</span>
                      <span className="font-mono text-foreground">{Number(val).toFixed(2)}</span>
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
        <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
          <span>Budget used</span>
          <span>{formatMoneyTenths((data?.squad?.totalCost) ?? null)} / £100.0</span>
        </div>
        <div className="h-2 w-full rounded-lg bg-muted overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-300"
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
            className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${filters[p as Position] ? "bg-primary text-primary-foreground border-primary" : "bg-background text-foreground border-border/50 hover:bg-accent hover:text-accent-foreground"}`}
          >
            {positionLabel(p as Position)}
          </button>
        ))}
      </div>

      {/* Club caps summary */}
      <div className="flex flex-wrap gap-1 text-xs">
        {clubCounts.map(([club, count]) => (
          <span key={club} className={`px-2 py-1 rounded-full border border-border/50 ${count >= 3 ? "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-200 dark:border-amber-800" : "text-muted-foreground bg-muted/30"}`}>
            {club}: {count}
          </span>
        ))}
      </div>

      {/* XI */}
      <div className="flex-1 min-h-0 overflow-auto rounded-lg border border-border/50 bg-background/50">
        <table className="w-full text-sm">
          <tbody>
            <tr className="font-mono text-xs text-muted-foreground uppercase text-center"><td className="p-3 border-b border-border/30" colSpan={99}>XI</td></tr>
            {filteredXi.map((p, idx) => (
              <tr key={`xi-${idx}`} className="hover:bg-muted/40 transition-colors border-b border-border/20 last:border-b-0">
                <td className="px-3 py-2 font-medium flex items-center gap-2">
                  {p.name}
                  {p.id === captainId && <span className="text-xs px-1.5 py-0.5 rounded-md bg-primary text-primary-foreground font-semibold">C</span>}
                  {p.id === viceId && <span className="text-xs px-1.5 py-0.5 rounded-md bg-muted text-muted-foreground font-semibold">VC</span>}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground">{positionLabel(p.position)}</td>
                <td className="px-3 py-2 text-xs">{p.team.shortName}</td>
                <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{formatMoneyTenths(p.cost)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs text-primary">{p.expectedPoints?.toFixed(1) ?? "-"}</td>
                <td className="px-3 py-2 text-right">
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
                <td className="p-4 text-center text-xs text-muted-foreground" colSpan={99}>
                  No XI players match current filters.
                </td>
              </tr>
            )}
            <tr className="font-mono text-xs text-muted-foreground uppercase text-center"><td className="p-3 border-t border-border/30" colSpan={99}>Bench</td></tr>
            {filteredBench.map((p, idx) => (
              <tr key={`bench-${idx}`} className="hover:bg-muted/40 transition-colors border-b border-border/20 last:border-b-0">
                <td className="px-3 py-2 font-medium">{p.name}</td>
                <td className="px-3 py-2 text-xs text-muted-foreground">{positionLabel(p.position)}</td>
                <td className="px-3 py-2 text-xs">{p.team.shortName}</td>
                <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{formatMoneyTenths(p.cost)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs text-primary">{p.expectedPoints?.toFixed(1) ?? "-"}</td>
                <td className="px-3 py-2 text-right">
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
                <td className="p-4 text-center text-xs text-muted-foreground" colSpan={99}>
                  No bench players match current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Totals */}
      <div className="flex justify-end text-primary text-sm bg-muted/20 rounded-lg p-3">
        <div className="mr-2 text-muted-foreground">Expected Points (bench boost):</div>
        <div className="font-mono font-semibold">
          {predictedPoints.toFixed(1)} ({benchBoostPredictedPoints.toFixed(1)})
        </div>
      </div>
    </div>
  );
}
