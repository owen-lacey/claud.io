// @ts-nocheck
import { test, expect } from '@playwright/test';

// E2E: build squad via chat → panel updates → explain picks

test.describe('Assistant flow', () => {
  test('build squad -> panel updates -> explain picks', async ({ page }) => {
    // Mock backend API the UI calls for normalization/context
    await page.route('http://localhost:5079/players', async (route) => {
      const players = [
        {
          id: 1, position: 1, cost: 45, chanceOfPlayingNextRound: 100,
          firstName: 'Test', secondName: 'GK', xp: 3.2, selectedByPercent: 10,
          team: 1, seasonPoints: 0, yellowCards: 0, redCards: 0, transfersOut: 0,
          name: 'Test GK', isAvailable: true, predictions: {}
        },
        {
          id: 2, position: 2, cost: 50, chanceOfPlayingNextRound: 100,
          firstName: 'Test', secondName: 'DEF', xp: 4.5, selectedByPercent: 5,
          team: 1, seasonPoints: 0, yellowCards: 0, redCards: 0, transfersOut: 0,
          name: 'Test DEF', isAvailable: true, predictions: {}
        }
      ];
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(players) });
    });
    await page.route('http://localhost:5079/teams', async (route) => {
      const teams = [ { id: 1, name: 'Arsenal', shortName: 'ARS', code: 1 } ];
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

    // Stub the tools-enabled chat stream with a simple payload line containing a selectedSquad
    await page.route('**/api/chat-tools', async (route) => {
      const streamLines = [
        '0: hello',
        'a: ' + JSON.stringify({ selectedSquad: {
          startingXi: [ { isCaptain: true, player: { id: 1, name: 'Test GK', position: 1, team: 1, cost: 45, xp: 3.2 } } ],
          bench: [ { player: { id: 2, name: 'Test DEF', position: 2, team: 1, cost: 50, xp: 4.5 } } ],
          squadCost: 95,
          predictedPoints: 7.7,
          benchBoostPredictedPoints: 0
        } }),
        '0: done'
      ].join('\n');
      await route.fulfill({ status: 200, contentType: 'text/plain; charset=utf-8', body: streamLines });
    });

    // Stub explain-selection tool API
    await page.route('**/api/tools/explain-selection', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
        notes: ['Fixture density is high; rotate bench.'],
        picks: [{ id: 1, reason: 'High save potential this week.' }]
      }) });
    });

    await page.addInitScript(() => {
      try { localStorage.setItem('pl_profile', 'test-token'); } catch {}
    });

    await page.goto('/assistant');

    // Type a prompt to trigger tool usage
    const textbox = page.getByRole('textbox');
    await textbox.fill('Build a wildcard squad for this week.');
    await textbox.press('Enter');

    // Wait for right panel header to switch to Chat Squad
    await expect(page.getByText('Chat Squad')).toBeVisible({ timeout: 30_000 });

    // Click Explain and wait for mocked API response
    const explainBtn = page.getByRole('button', { name: /Explain/i });
    await Promise.all([
      page.waitForResponse((res) => res.url().includes('/api/tools/explain-selection') && res.status() === 200),
      explainBtn.click(),
    ]);

    // After explain returns, expect our mocked note to show
    await expect(page.getByText('Fixture density is high; rotate bench.')).toBeVisible({ timeout: 10_000 });
  });
});
