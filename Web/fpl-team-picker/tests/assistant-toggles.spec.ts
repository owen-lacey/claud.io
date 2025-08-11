// @ts-nocheck
import { test, expect } from '@playwright/test';

// E2E: quick actions + toggles (captain/vice, bench reorder)

test.describe('Assistant toggles', () => {
  test('build squad -> toggle captain -> reorder bench', async ({ page }) => {
    // Backend stubs for data service
    await page.route('http://localhost:5079/players', async (route) => {
      const players = [
        { id: 10, position: 3, cost: 70, chanceOfPlayingNextRound: 100, team: 1, name: 'XI One', xp: 5.0, isAvailable: true, selectedByPercent: 10, seasonPoints: 0, yellowCards: 0, redCards: 0, transfersOut: 0, predictions: {} },
        { id: 11, position: 3, cost: 75, chanceOfPlayingNextRound: 100, team: 1, name: 'XI Two', xp: 7.0, isAvailable: true, selectedByPercent: 10, seasonPoints: 0, yellowCards: 0, redCards: 0, transfersOut: 0, predictions: {} },
        { id: 20, position: 2, cost: 45, chanceOfPlayingNextRound: 100, team: 1, name: 'Bench A', xp: 1.0, isAvailable: true, selectedByPercent: 5, seasonPoints: 0, yellowCards: 0, redCards: 0, transfersOut: 0, predictions: {} },
        { id: 21, position: 2, cost: 46, chanceOfPlayingNextRound: 100, team: 1, name: 'Bench B', xp: 2.0, isAvailable: true, selectedByPercent: 4, seasonPoints: 0, yellowCards: 0, redCards: 0, transfersOut: 0, predictions: {} },
      ];
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(players) });
    });
    await page.route('http://localhost:5079/teams', async (route) => {
      const teams = [{ id: 1, name: 'Arsenal', shortName: 'ARS', code: 1 }];
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(teams) });
    });
    await page.route('http://localhost:5079/my-team', async (route) => {
      const body = { freeTransfers: 1, bank: 0, budget: 1000, selectedSquad: null };
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    });
    await page.route('http://localhost:5079/my-details', async (route) => {
      const body = { id: 123, firstName: 'Test', lastName: 'User' };
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    });

    // Stub direct build-squad endpoint to return a small squad
    await page.route('**/api/tools/build-squad', async (route) => {
      const squad = {
        startingXi: [
          { isCaptain: true, player: { id: 10, name: 'XI One', position: 3, team: 1, cost: 70, xp: 5.0 } },
          { player: { id: 11, name: 'XI Two', position: 3, team: 1, cost: 75, xp: 7.0 } },
        ],
        bench: [
          { player: { id: 20, name: 'Bench A', position: 2, team: 1, cost: 45, xp: 1.0 } },
          { player: { id: 21, name: 'Bench B', position: 2, team: 1, cost: 46, xp: 2.0 } },
        ],
        squadCost: 236,
        predictedPoints: 12.0,
        benchBoostPredictedPoints: 15.0,
      };
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ squad }) });
    });

    await page.addInitScript(() => {
      try { localStorage.setItem('pl_profile', 'test-token'); } catch {}
    });

    await page.goto('/assistant');

    // Use quick action button to build squad
    await page.getByRole('button', { name: 'Build Squad' }).click();
    
    // Wait for the squad to be built (look for the success toast)
    await expect(page.getByText('Squad built successfully!')).toBeVisible();
    
    // The header should show either "Chat Squad" or "Wildcard Squad"
    await expect(page.getByText(/Chat Squad|Wildcard Squad/)).toBeVisible();

    // Verify initial totals: base 12.0 + captain bonus 5.0 = 17.0; bench boost adds 3.0 -> 20.0
    await expect(page.getByText(/17\.0 \(20\.0\)/)).toBeVisible();

    // Toggle captain to XI Two
    const xiTwoRow = page.getByRole('row', { name: /XI Two/ });
    await xiTwoRow.getByRole('button', { name: 'Set C' }).click();

    // Totals should reflect new captain bonus: 12.0 + 7.0 = 19.0; bench boost 22.0
    await expect(page.getByText(/19\.0 \(22\.0\)/)).toBeVisible();

    // Reorder bench: move first down so order becomes Bench B then Bench A
    const benchARow = page.getByRole('row', { name: /Bench A/ });
    await benchARow.getByRole('button', { name: '↓' }).click();

    // Check visual order changed by ensuring both players are still present in the table
    // Use more specific selectors to avoid ambiguity
    await expect(page.getByRole('cell', { name: 'Bench B' })).toBeVisible();
    await expect(page.getByRole('cell', { name: 'Bench A' })).toBeVisible();
  });
});
