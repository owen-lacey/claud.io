using FluentAssertions;
using FluentAssertions.Execution;
using FplTeamPicker.Optimisation;
using FplTeamPicker.Optimisation.Models;
using FplTeamPicker.Tests.Builders;

namespace FplTeamPicker.Tests;

public class FplTeamTransfersSolverTests
{
    [Fact]
    public void BetterPlayerIsTooExpensive_NotTransferredIn()
    {
        var existingTeam = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithCost(25).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithCost(25).Build()
        };
        var expensivePlayer = new FplPlayerBuilder("LIV", PlayerPosition.DEF)
            .WithPredictedPoints(100)
            .WithCost(26)
            .Build();
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 100,
            StartingTeamCount = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, [expensivePlayer], options, 1, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.PlayersOut.Should().BeEmpty();
        }
    }

    [Fact]
    public void BetterPlayerWouldExceedMaxPerTeamRule_NotTransferredIn()
    {
        var existingTeam = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).Build(),
            new FplPlayerBuilder("MNU", PlayerPosition.FWD).Build()
        };
        var expensivePlayer = new FplPlayerBuilder("LIV", PlayerPosition.FWD)
            .WithPredictedPoints(100)
            .Build();
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 3,
            StartingTeamCount = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, [expensivePlayer], options, 1, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.PlayersOut.Should().BeEmpty();
            output.PlayersIn.Should().BeEmpty();
            output.Squad.Should().HaveCount(existingTeam.Count);
        }
    }

    [Fact]
    public void BetterPlayerPlaysInDifferentPosition_NotTransferredIn()
    {
        var existingTeam = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithPredictedPoints(98).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(100).Build()
        };
        var expensivePlayer = new FplPlayerBuilder("LIV", PlayerPosition.FWD)
            .WithPredictedPoints(99)
            .Build();
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 100,
            StartingTeamCount = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, [expensivePlayer], options, 1, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.PlayersOut.Should().BeEmpty();
            output.PlayersIn.Should().BeEmpty();
            output.Squad.Should().HaveCount(existingTeam.Count);
        }
    }

    [Fact]
    public void NoBetterPlayersAvailable_NotTransferredIn()
    {
        var existingTeam = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(100).Build()
        };
        var existingPlayers = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithPredictedPoints(99).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithPredictedPoints(99).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithPredictedPoints(99).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(99).Build()
        };
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 100,
            StartingTeamCount = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, existingPlayers, options, 4, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.PlayersOut.Should().BeEmpty();
            output.PlayersIn.Should().BeEmpty();
            output.Squad.Should().HaveCount(existingTeam.Count);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BetterPlayersAvailable_TransferredIn(int freeTransfers)
    {
        var existingTeam = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(100).Build()
        };
        var existingPlayers = new List<FplPlayer>
        {
            new FplPlayerBuilder("LIV", PlayerPosition.GK).WithPredictedPoints(101).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.DEF).WithPredictedPoints(101).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.MID).WithPredictedPoints(101).Build(),
            new FplPlayerBuilder("LIV", PlayerPosition.FWD).WithPredictedPoints(101).Build()
        };
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 100,
            StartingTeamCount = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, existingPlayers, options, freeTransfers, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        using (new AssertionScope())
        {
            output.PlayersIn.Should().HaveCount(freeTransfers);
            output.PlayersOut.Should().HaveCount(freeTransfers);
            output.Squad.Should().HaveCount(existingTeam.Count);
        }
    }
}