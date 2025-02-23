using FluentAssertions;
using FluentAssertions.Execution;
using FplTeamPicker.Models;
using FplTeamPicker.Tests.Builders;

namespace FplTeamPicker.Tests;

public class PickFplTeamSolverTests
{
    [Fact]
    public void LotsOfGoodPlayersPlayForTeam_PicksWorsePlayersToNotExceedMaxPlayersRule()
    {
        var goodPlayers = Enumerable.Range(0, 3)
            .Select(_ => new FplPlayerBuilder("LIV", PlayerPosition.DEF)
                .WithPredictedPoints(100)
                .Build())
            .ToList();
        var badPlayers = Enumerable.Range(0, 3)
            .Select(_ => new FplPlayerBuilder("MNU", PlayerPosition.DEF).Build())
            .ToList();
        var players = goodPlayers
            .Concat(badPlayers)
            .Concat([new FplPlayerBuilder("MCI", PlayerPosition.GK).Build()])
            .ToList();
        var options = new FplOptions
        {
            SquadGoalkeeperCount = 1,
            SquadDefenderCount = 3,
            MaxPlayersPerTeam = 2,
            StartingTeamCount = 4
        };
        const int budget = 10;
        var model = new PickFplTeamModel(players, options, budget);
        var solver = new PickFplTeamSolver(model);

        var output = solver.Solve();

        output.Squad.Should().Contain(
            p => p.Team == "MNU",
            $"we can only have {options.MaxPlayersPerTeam} max players per team");
    }

    [Fact]
    public void LotsOfGoodPlayersForSpecificPosition_PicksWorsePlayersToFillQuota()
    {
        var players = new List<FplPlayer>
        {
            new FplPlayerBuilder(Guid.NewGuid().ToString(), PlayerPosition.GK).Build(),
            new FplPlayerBuilder(Guid.NewGuid().ToString(), PlayerPosition.DEF).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(Guid.NewGuid().ToString(), PlayerPosition.DEF).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(Guid.NewGuid().ToString(), PlayerPosition.MID).WithPredictedPoints(99).Build()
        };
        var options = new FplOptions
        {
            SquadGoalkeeperCount = 1,
            SquadDefenderCount = 1,
            SquadMidfielderCount = 1,
            StartingTeamCount = 3
        };
        const int budget = 10;
        var model = new PickFplTeamModel(players, options, budget);
        var solver = new PickFplTeamSolver(model);

        var output = solver.Solve();

        output.Squad.Should().Contain(
            p => p.Position == PlayerPosition.MID,
            $"we can only have {options.SquadDefenderCount} defenders in our team");
    }

    [Fact]
    public void GoodPlayerIsTooExpensive_PicksWorsePlayersToNotExceedBudget()
    {
        var expensivePlayer = new FplPlayerBuilder("LIV", PlayerPosition.FWD)
            .WithCost(26)
            .WithPredictedPoints(100)
            .Build();
        var players = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithCost(25).Build(),
            expensivePlayer,
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithCost(25).Build()
        };
        const int budget = 100;
        var options = new FplOptions
        {
            SquadGoalkeeperCount = 1,
            SquadDefenderCount = 1,
            SquadMidfielderCount = 1,
            SquadForwardCount = 1,
            MaxPlayersPerTeam = players.Count,
            StartingTeamCount = 4
        };
        var model = new PickFplTeamModel(players, options, budget);
        var solver = new PickFplTeamSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.Squad.Should().HaveCountGreaterThan(0, "we should have at least one player");
            output.Squad.Should().NotContain(
                expensivePlayer,
                $"we cannot afford to have the better player as it would exceed the budget of {budget}");
        }
    }

    [Fact]
    public void PlayersAreWithinBudget_MaximisesPlayerSelection()
    {
        var badForward = new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(49).WithCost(25).Build();
        var goodForward = new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(51).WithCost(25).Build();
        var players = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithCost(25).Build(),
            goodForward,
            badForward
        };
        const int budget = 100;
        var options = new FplOptions
        {
            SquadGoalkeeperCount = 1,
            SquadDefenderCount = 1,
            SquadMidfielderCount = 1,
            SquadForwardCount = 1,
            MaxPlayersPerTeam = players.Count,
            StartingTeamCount = 4
        };
        var model = new PickFplTeamModel(players, options, budget);
        var solver = new PickFplTeamSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.Squad.Should().HaveCountGreaterThan(0, "we should have at least one player");
            output.Squad.Should().NotContain(
                badForward,
                "we should not select the player with the lower selection percentage");
            output.Squad.Should().Contain(
                goodForward,
                "we should select the player that has a higher selection percentage");
        }
    }
}