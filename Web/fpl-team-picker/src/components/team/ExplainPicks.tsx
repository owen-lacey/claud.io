"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

type ExplainResult = {
  notes: string[];
  picks: Array<{ id: number; reason: string }>;
};

export default function ExplainPicks({ squad }: { squad: any | null }) {
  const [result, setResult] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function flattenSquad(input: any | null) {
    if (!input) return null;
    const looksBackend = Array.isArray(input.startingXi) && input.startingXi[0] && input.startingXi[0].player;
    if (!looksBackend) return input;
    const toFlat = (arr: any[]) => arr.map((sp) => ({
      id: sp.player?.id,
      name: sp.player?.name,
      position: sp.player?.position,
      team: sp.player?.team,
      cost: sp.player?.cost,
      expectedPoints: sp.player?.xp ?? sp.player?.expectedPoints,
    }));
    return {
      startingXi: toFlat(input.startingXi || []),
      bench: toFlat(input.bench || []),
      totalCost: input.squadCost,
      predictedPoints: input.predictedPoints,
    };
  }

  const run = async () => {
    if (!squad) return;
    setLoading(true);
    setError(null);
    try {
      const flat = flattenSquad(squad);
      const res = await fetch("/api/tools/explain-selection", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ squad: flat }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Failed to explain");
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? "Failed to explain");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setResult(null);
  }, [squad]);

  const hasContent = result && (result.notes?.length || result.picks?.length);

  return (
    <div className="border rounded-md p-3 flex flex-col gap-3 bg-card">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Explain Picks</h2>
          <p className="text-xs text-muted-foreground">Generate concise rationale</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={run} disabled={!squad || loading}>
            {loading ? "Explaining..." : "Explain"}
          </Button>
        </div>
      </div>

      {error && <div className="text-sm text-destructive">{error}</div>}

      {/* Loading skeleton */}
      {loading && (
        <div className="grid gap-2 animate-pulse">
          <div className="h-3 w-2/3 rounded bg-muted" />
          <div className="h-24 rounded border bg-background/50" />
          <div className="h-24 rounded border bg-background/50" />
        </div>
      )}

      {!loading && !result && (
        <div className="text-sm text-muted-foreground">
          {squad ? "Click Explain to summarize this squad." : "No squad to explain."}
        </div>
      )}

      {!loading && result && !hasContent && (
        <div className="text-sm text-muted-foreground">No notes or picks returned.</div>
      )}

      {!loading && hasContent && (
        <div className="flex flex-col gap-2 text-sm">
          {result!.notes.length > 0 && (
            <ul className="list-disc pl-5 text-muted-foreground">
              {result!.notes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          )}
          {result!.picks.length > 0 && (
            <div className="grid gap-1">
              {result!.picks.map((p, i) => (
                <div key={i} className="rounded-md border bg-background/50 p-2">
                  <span className="font-mono text-xs mr-2">#{p.id}</span>
                  <span>{p.reason}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
