import { FPL_RULES, addToCounts, emptyCounts, isValidFinalSquad, isValidPartialSquad } from './constraints';
import { Position } from '@/models/position';
import type { NormalizedPlayer, NormalizedSquad } from '@/lib/data-service';
import { comparePlayers, defaultScoringOptions, type ScoringOptions, scorePlayer } from './scoring';

export type SelectionOptions = {
  scoring?: ScoringOptions;
};

export type SelectionResult = {
  squad: NormalizedSquad;
  explanation: {
    notes: string[];
    picks: Array<{ playerId: number; reason: string; score: number }>;
  };
};

function byPosition(players: NormalizedPlayer[]) {
  return {
    [Position.GK]: players.filter(p => p.position === Position.GK),
    [Position.DEF]: players.filter(p => p.position === Position.DEF),
    [Position.MID]: players.filter(p => p.position === Position.MID),
    [Position.FWD]: players.filter(p => p.position === Position.FWD),
  } as Record<Position, NormalizedPlayer[]>;
}

export function buildSquad(
  pool: NormalizedPlayer[],
  budget: number,
  opts: SelectionOptions = {}
): SelectionResult {
  const scoring = opts.scoring ?? defaultScoringOptions;
  const grouped = byPosition(pool);

  // sort each position bucket by score deterministically
  (Object.keys(grouped) as unknown as Position[]).forEach(pos => {
    grouped[pos] = grouped[pos].slice().sort((a, b) => comparePlayers(a, b, scoring));
  });

  const quotas = FPL_RULES.positionQuotas;
  let counts = emptyCounts();
  const picked: NormalizedPlayer[] = [];
  const explanationPicks: Array<{ playerId: number; reason: string; score: number }> = [];
  let totalCost = 0;

  function tryPick(pos: Position, needed: number) {
    for (const cand of grouped[pos]) {
      if (picked.includes(cand)) continue;
      // budget check first
      if (totalCost + cand.cost > budget) continue;
      // simulate counts if we add this candidate
      const nextCounts = addToCounts(counts, cand);
      if (!isValidPartialSquad(nextCounts)) continue;
      // accept pick
      picked.push(cand);
      counts = nextCounts;
      totalCost += cand.cost;
      explanationPicks.push({ playerId: cand.id, reason: `Top ${Position[pos]} option by score within budget`, score: scorePlayer(cand, scoring) });
      if (needed > 1) {
        if (picked.filter(p => p.position === pos).length >= (FPL_RULES.positionQuotas as any)[Position[pos]]) break;
      }
      if (picked.filter(p => p.position === pos).length >= needed) break;
    }
  }

  // Greedy fill by quotas order
  tryPick(Position.GK, quotas.GK);
  tryPick(Position.DEF, quotas.DEF);
  tryPick(Position.MID, quotas.MID);
  tryPick(Position.FWD, quotas.FWD);

  // If not full due to tight constraints, attempt to fill remaining slots from any position with remaining quota
  while (picked.length < FPL_RULES.squadSize) {
    let added = false;
    for (const pos of [Position.FWD, Position.MID, Position.DEF, Position.GK]) {
      const current = picked.filter(p => p.position === pos).length;
      const quota = (quotas as any)[Position[pos]] as number;
      if (current >= quota) continue;
      for (const cand of grouped[pos]) {
        if (picked.includes(cand)) continue;
        if (totalCost + cand.cost > budget) continue;
        const nextCounts = addToCounts(counts, cand);
        if (!isValidPartialSquad(nextCounts)) continue;
        picked.push(cand);
        counts = nextCounts;
        totalCost += cand.cost;
        explanationPicks.push({ playerId: cand.id, reason: `Filled remaining slot at ${Position[pos]}` , score: scorePlayer(cand, scoring)});
        added = true;
        break;
      }
      if (added) break;
    }
    if (!added) break;
  }

  // Build NormalizedSquad (simple version: first 11 as starting XI by score, rest bench)
  const sortedPicked = picked.slice().sort((a, b) => comparePlayers(a, b, scoring));
  const startingXi = sortedPicked.slice(0, Math.min(11, sortedPicked.length));
  const bench = sortedPicked.slice(11);

  const squad: NormalizedSquad = {
    startingXi,
    bench,
    captain: startingXi[0] ?? null,
    viceCaptain: startingXi[1] ?? null,
    totalCost,
    predictedPoints: startingXi.reduce((sum, p) => sum + (p.expectedPoints ?? 0), 0),
    benchBoostPredictedPoints: bench.reduce((sum, p) => sum + (p.expectedPoints ?? 0), 0),
  };

  const notes: string[] = [];
  notes.push(`Budget used: ${(totalCost/10).toFixed(1)}m of ${(budget/10).toFixed(1)}m`);
  notes.push(`Constraints respected: ${isValidFinalSquad(counts, totalCost, budget - totalCost) ? 'Yes' : 'Partially (check per-club/quotas)'}`);

  return {
    squad,
    explanation: {
      notes,
      picks: explanationPicks,
    },
  };
}
