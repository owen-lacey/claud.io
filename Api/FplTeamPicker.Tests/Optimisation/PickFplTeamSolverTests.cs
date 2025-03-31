using FluentAssertions;
using FluentAssertions.Execution;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using FplTeamPicker.Tests.Optimisation.Builders;

namespace FplTeamPicker.Tests.Optimisation;

public class PickFplTeamSolverTests
{
    [Fact]
    public void LotsOfGoodPlayersPlayForTeam_PicksWorsePlayersToNotExceedMaxPlayersRule()
    {
        const int teamOne = 1;
        const int teamTwo = 2;
        var goodPlayers = Enumerable.Range(0, 3)
            .Select(_ => new FplPlayerBuilder(teamOne, Position.Defender)
                .WithPredictedPoints(100)
                .Build())
            .ToList();
        var badPlayers = Enumerable.Range(0, 3)
            .Select(_ => new FplPlayerBuilder(teamTwo, Position.Defender).Build())
            .ToList();
        var players = goodPlayers
            .Concat(badPlayers)
            .Concat(new List<Player> { new FplPlayerBuilder(teamTwo, Position.Goalkeeper).Build() })
            .OrderBy(r => r.Id)
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

        output.StartingXi.Should().Contain(
            p => p.Player.Team == teamTwo,
            $"we can only have {options.MaxPlayersPerTeam} max players per team");
    }

    [Fact]
    public void LotsOfGoodPlayersForSpecificPosition_PicksWorsePlayersToFillQuota()
    {
        var players = new List<Player>
            {
                new FplPlayerBuilder(1, Position.Goalkeeper).Build(),
                new FplPlayerBuilder(2, Position.Defender).WithPredictedPoints(100).Build(),
                new FplPlayerBuilder(3, Position.Defender).WithPredictedPoints(100).Build(),
                new FplPlayerBuilder(4, Position.Defender).WithPredictedPoints(100).Build(),
                new FplPlayerBuilder(5, Position.Defender).WithPredictedPoints(100).Build(),
                new FplPlayerBuilder(6, Position.Midfielder).Build()
            }
            .OrderBy(r => r.Id)
            .ToList();
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

        output.StartingXi.Count(p => p.Player.Position == Position.Defender).Should().Be(
            1,
            $"we can only have {options.SquadDefenderCount} defenders in our team");
    }

    [Fact]
    public void GoodPlayerIsTooExpensive_PicksWorsePlayersToNotExceedBudget()
    {
        const int teamOne = 1;
        var expensivePlayer = new FplPlayerBuilder(teamOne, Position.Forward)
            .WithCost(26)
            .WithPredictedPoints(100)
            .Build();
        var players = new List<Player>
            {
                new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithCost(25).Build(),
                new FplPlayerBuilder(teamOne, Position.Defender).WithCost(25).Build(),
                new FplPlayerBuilder(teamOne, Position.Midfielder).WithCost(25).Build(),
                expensivePlayer,
                new FplPlayerBuilder(teamOne, Position.Forward).WithCost(25).Build()
            }
            .OrderBy(r => r.Id)
            .ToList();
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
            output.StartingXi.Should().HaveCountGreaterThan(0, "we should have at least one player");
            output.StartingXi.Any(p => p.Player.Id == expensivePlayer.Id).Should().BeFalse(
                $"we cannot afford to have the better player as it would exceed the budget of {budget}");
        }
    }

    [Fact]
    public void PlayersAreWithinBudget_MaximisesPlayerSelection()
    {
        const int teamOne = 1;
        var badForward = new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(49).WithCost(25).Build();
        var goodForward = new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(51).WithCost(25).Build();
        var players = new List<Player>
            {
                new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithCost(25).Build(),
                new FplPlayerBuilder(teamOne, Position.Defender).WithCost(25).Build(),
                new FplPlayerBuilder(teamOne, Position.Midfielder).WithCost(25).Build(),
                goodForward,
                badForward
            }
            .OrderBy(r => r.Id)
            .ToList();
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
            output.StartingXi.Should().HaveCountGreaterThan(0, "we should have at least one player");
            output.StartingXi.Any(p => p.Player.Id == badForward.Id).Should().BeFalse(
                "we should not select the player with the lower selection percentage");
            output.StartingXi.Any(p => p.Player.Id == goodForward.Id).Should().BeTrue(
                "we should select the player that has a higher selection percentage");
        }
    }

    [Fact]
    public void PlayerThatWouldBeOnBenchIsCheaper_PicksCheaperPlayer()
    {
        const int teamOne = 1;
        var cheaperBenchPlayer = new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(0).WithCost(2).Build();
        var expensiveBenchPlayer = new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(0).WithCost(3).Build();
        var teamPlayer = new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(1).Build();
        var players = new List<Player>
            {
                new FplPlayerBuilder(teamOne, Position.Goalkeeper).Build(),
                expensiveBenchPlayer,
                cheaperBenchPlayer,
                teamPlayer,
                new FplPlayerBuilder(teamOne, Position.Midfielder).Build(),
                new FplPlayerBuilder(teamOne, Position.Forward).Build()
            }
            .OrderBy(r => r.Id)
            .ToList();
        const int budget = 1000;
        var options = new FplOptions
        {
            SquadGoalkeeperCount = 1,
            SquadDefenderCount = 2,
            SquadMidfielderCount = 1,
            SquadForwardCount = 1,
            MinTeamDefenders = 1,
            MinTeamForwards = 1,
            MinTeamMidfielders = 1,
            MaxPlayersPerTeam = players.Count,
            StartingTeamCount = 4
        };
        var model = new PickFplTeamModel(players, options, budget);
        var solver = new PickFplTeamSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            var teamDefender = output.StartingXi.Single(p => p.Player.Position == Position.Defender);
            var benchDefender = output.Bench.Single(p => p.Player.Position == Position.Defender);
            teamDefender.Player.Id.Should().Be(teamPlayer.Id, "we should pick the defender with some predicted points");
            benchDefender.Player.Id.Should().Be(cheaperBenchPlayer.Id, "the benched defender should be the cheapest");
            benchDefender.Player.Id.Should().NotBe(expensiveBenchPlayer.Id, "we should not pick expensive subs for the sake of it");
        }
    }
}