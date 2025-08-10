import type { NormalizedPlayer } from '@/lib/data-service';

export type ScoringOptions = {
  expectedPointsWeight: number; // weight for expected points (xp)
  availabilityWeight: number;   // additional lift for availability (0-1 based on chance)
  priceWeight: number;          // price efficiency weight (default 0 = ignore price)
};

export const defaultScoringOptions: ScoringOptions = {
  expectedPointsWeight: 1,
  availabilityWeight: 0.2,
  priceWeight: 0, // per user preference: only enforce budget, don't score on price
};

export function scorePlayer(p: NormalizedPlayer, opts: ScoringOptions = defaultScoringOptions): number {
  const xp = p.expectedPoints ?? 0;
  const availabilityFactor = (p.availability.chanceOfPlaying ?? 100) / 100; // 0..1
  const availabilityScore = opts.availabilityWeight * availabilityFactor;
  const priceScore = opts.priceWeight === 0 ? 0 : opts.priceWeight * (xp / Math.max(p.cost, 1));
  return opts.expectedPointsWeight * xp + availabilityScore + priceScore;
}

export function comparePlayers(a: NormalizedPlayer, b: NormalizedPlayer, opts: ScoringOptions = defaultScoringOptions): number {
  const sa = scorePlayer(a, opts);
  const sb = scorePlayer(b, opts);
  if (sb !== sa) return sb - sa; // higher score first
  // deterministic tie-breakers: availability desc, then id asc
  const availA = (a.availability.chanceOfPlaying ?? 100);
  const availB = (b.availability.chanceOfPlaying ?? 100);
  if (availB !== availA) return availB - availA;
  return a.id - b.id;
}
