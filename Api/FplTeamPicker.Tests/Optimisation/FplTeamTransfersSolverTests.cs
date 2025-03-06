using FluentAssertions;
using FluentAssertions.Execution;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using FplTeamPicker.Tests.Optimisation.Builders;

namespace FplTeamPicker.Tests.Optimisation;

public class FplTeamTransfersSolverTests
{
    [Fact]
    public void BetterPlayerIsTooExpensive_NotTransferredIn()
    {
        const int teamOne = 1;
        var existingTeam = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithCost(25).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithCost(25).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithCost(25).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithCost(25).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var expensivePlayer = new FplPlayerBuilder(teamOne, Position.Defender)
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
            output.StartingXi.Select(r => r.Player.Id).Should().BeEquivalentTo(existingTeam.Select(t => t.Id));
        }
    }

    [Fact]
    public void BetterPlayerWouldExceedMaxPerTeamRule_NotTransferredIn()
    {
        const int teamOne = 1;
        const int teamTwo = 2;
        var existingTeam = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).Build(),
            new FplPlayerBuilder(teamTwo, Position.Forward).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var expensivePlayer = new FplPlayerBuilder(teamOne, Position.Forward)
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
            output.StartingXi.Select(r => r.Player.Id).Should().BeEquivalentTo(existingTeam.Select(t => t.Id));
            output.StartingXi.Should().HaveCount(existingTeam.Count);
        }
    }

    [Fact]
    public void BetterPlayerPlaysInDifferentPosition_NotTransferredIn()
    {
        const int teamOne = 1;
        var existingTeam = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithPredictedPoints(98).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(100).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var expensivePlayer = new FplPlayerBuilder(teamOne, Position.Forward)
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
            output.StartingXi.Select(r => r.Player.Id).Should().BeEquivalentTo(existingTeam.Select(t => t.Id));
            output.StartingXi.Should().HaveCount(existingTeam.Count);
        }
    }

    [Fact]
    public void NoBetterPlayersAvailable_NotTransferredIn()
    {
        const int teamOne = 1;
        var existingTeam = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(100).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var existingPlayers = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithPredictedPoints(99).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(99).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithPredictedPoints(99).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(99).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
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
            output.StartingXi.Select(r => r.Player.Id).Should().BeEquivalentTo(existingTeam.Select(t => t.Id));
            output.StartingXi.Should().HaveCount(existingTeam.Count);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BetterPlayersAvailable_TransferredIn(int freeTransfers)
    {
        const int teamOne = 1;
        var existingTeam = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(100).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var existingPlayers = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithPredictedPoints(101).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(101).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithPredictedPoints(101).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(101).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 100,
            StartingTeamCount = 4,
            TransferPointsPenalty = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, existingPlayers, options, freeTransfers, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        var playersIn = output.StartingXi.Where(r => existingTeam.All(p => p.Id != r.Player.Id)).ToList();
        var playersOut = existingTeam.Where(r => output.StartingXi.All(p => p.Player.Id != r.Id)).ToList();
        using (new AssertionScope())
        {
            playersIn.Should().HaveCount(freeTransfers);
            playersOut.Should().HaveCount(freeTransfers);
        }
    }

    [Theory]
    [InlineData(104, true)]
    [InlineData(103, false)]
    public void BetterPlayerWouldExceedFreeTransfers_OnlyTransfersIfSignificantlyBetter(decimal expectedPoints, bool shouldTransfer)
    {
        const int teamOne = 1;
        var existingTeam = new List<Player>
        {
            new FplPlayerBuilder(teamOne, Position.Goalkeeper).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Defender).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Midfielder).WithPredictedPoints(100).Build(),
            new FplPlayerBuilder(teamOne, Position.Forward).WithPredictedPoints(100).Build()
        }
        .OrderBy(r => r.Id)
        .ToList();
        var betterPlayer = new FplPlayerBuilder(teamOne, Position.Forward)
            .WithPredictedPoints(expectedPoints)
            .Build();
        var options = new FplOptions
        {
            MaxPlayersPerTeam = 100,
            StartingTeamCount = 4,
            TransferPointsPenalty = 4
        };
        var model = new FplTeamTransfersRequest(existingTeam, [betterPlayer], options, 0, 0);
        var solver = new FplTeamTransfersSolver(model);

        var output = solver.Solve();

        var playersIn = output.StartingXi.Where(r => existingTeam.All(p => p.Id != r.Player.Id)).ToList();
        var playersOut = existingTeam.Where(r => output.StartingXi.All(p => p.Player.Id != r.Id)).ToList();
        using (new AssertionScope())
        {
            if (shouldTransfer)
            {
                playersIn.Should().HaveCount(1);
                playersOut.Should().HaveCount(1);
            }
            else
            {
                playersIn.Should().BeEmpty();
                playersOut.Should().BeEmpty();
            }
        }
    }
}