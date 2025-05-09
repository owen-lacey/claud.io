using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Models;

namespace FplTeamPicker.Services.Optimisation.UseCases.Transfers;

public record TransfersModelInput
{
    public TransfersModelInput(
        List<Player> existingPlayers,
        List<Player> otherPlayers,
        FplOptions options,
        int numberTransfers,
        int remainingBudget)
    {
        Options = options;
        NumberTransfers = numberTransfers;
        RemainingBudget = remainingBudget;
        ExistingPlayers = existingPlayers;
        OtherPlayers = otherPlayers;

        Options.SquadForwardCount = existingPlayers.Count(p => p.Position == Position.Forward);
        Options.SquadMidfielderCount = existingPlayers.Count(p => p.Position == Position.Midfielder);
        Options.SquadDefenderCount = existingPlayers.Count(p => p.Position == Position.Defender);
        Options.SquadGoalkeeperCount = existingPlayers.Count(p => p.Position == Position.Goalkeeper);
    }

    public int NumberTransfers { get; }

    public decimal RemainingBudget { get; }

    public List<Player> OtherPlayers { get; }

    public List<Player> ExistingPlayers { get; }

    public FplOptions Options { get; }

    public List<Player> AllPlayers => ExistingPlayers.Union(OtherPlayers)
        .OrderBy(r => r.Id)
        .ToList();

    public int Budget => ExistingPlayers.Sum(e => e.Cost) + (int)Math.Round(RemainingBudget);
}