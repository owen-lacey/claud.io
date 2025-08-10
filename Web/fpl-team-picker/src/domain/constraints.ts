import { Position } from '@/models/position';
import { NormalizedPlayer } from '@/lib/data-service';

export type SquadCounts = {
  GK: number;
  DEF: number;
  MID: number;
  FWD: number;
  perTeam: Record<number, number>; // teamId -> count
  total: number;
};

export const FPL_RULES = {
  budgetCap: 1000, // 100.0m represented in tenths
  squadSize: 15,
  positionQuotas: {
    GK: 2,
    DEF: 5,
    MID: 5,
    FWD: 3,
  },
  maxPerClub: 3,
};

export function emptyCounts(): SquadCounts {
  return { GK: 0, DEF: 0, MID: 0, FWD: 0, perTeam: {}, total: 0 };
}

export function addToCounts(counts: SquadCounts, p: NormalizedPlayer): SquadCounts {
  const next = { ...counts, perTeam: { ...counts.perTeam } };
  next.total += 1;
  switch (p.position) {
    case Position.GK: next.GK += 1; break;
    case Position.DEF: next.DEF += 1; break;
    case Position.MID: next.MID += 1; break;
    case Position.FWD: next.FWD += 1; break;
  }
  next.perTeam[p.team.id] = (next.perTeam[p.team.id] || 0) + 1;
  return next;
}

export function withinBudget(totalCost: number, bank: number, cap: number = FPL_RULES.budgetCap): boolean {
  return totalCost <= cap && bank >= 0;
}

export function respectsPositionQuotas(counts: SquadCounts, quotas = FPL_RULES.positionQuotas): boolean {
  return counts.GK <= quotas.GK && counts.DEF <= quotas.DEF && counts.MID <= quotas.MID && counts.FWD <= quotas.FWD;
}

export function respectsMaxPerClub(counts: SquadCounts, maxPerClub = FPL_RULES.maxPerClub): boolean {
  return Object.values(counts.perTeam).every((n) => n <= maxPerClub);
}

export function isValidPartialSquad(counts: SquadCounts): boolean {
  return respectsPositionQuotas(counts) && respectsMaxPerClub(counts) && counts.total <= FPL_RULES.squadSize;
}

export function isValidFinalSquad(counts: SquadCounts, totalCost: number, bank: number): boolean {
  return (
    counts.total === FPL_RULES.squadSize &&
    counts.GK === FPL_RULES.positionQuotas.GK &&
    counts.DEF === FPL_RULES.positionQuotas.DEF &&
    counts.MID === FPL_RULES.positionQuotas.MID &&
    counts.FWD === FPL_RULES.positionQuotas.FWD &&
    respectsMaxPerClub(counts) &&
    withinBudget(totalCost, bank)
  );
}

// Captaincy rules placeholder (expand in later milestone)
export function validCaptaincy(captainId: number | null, viceCaptainId: number | null, squad: NormalizedPlayer[]): boolean {
  if (captainId == null || viceCaptainId == null) return true; // not enforced yet
  if (captainId === viceCaptainId) return false;
  const ids = new Set(squad.map((p) => p.id));
  return ids.has(captainId) && ids.has(viceCaptainId);
}
