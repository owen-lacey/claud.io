namespace FplTeamPicker.Optimisation.Models;

public record FplTeamTransfersRequest
{
    public FplTeamTransfersRequest(
        List<FplPlayer> existingPlayers,
        List<FplPlayer> otherPlayers,
        FplOptions options,
        int numberTransfers,
        decimal remainingBudget)
    {
        Options = options;
        NumberTransfers = numberTransfers;
        RemainingBudget = remainingBudget;
        ExistingPlayers = existingPlayers;
        OtherPlayers = otherPlayers;

        Options.SquadForwardCount = existingPlayers.Count(p => p.Position == PlayerPosition.FWD);
        Options.SquadMidfielderCount = existingPlayers.Count(p => p.Position == PlayerPosition.MID);
        Options.SquadDefenderCount = existingPlayers.Count(p => p.Position == PlayerPosition.DEF);
        Options.SquadGoalkeeperCount = existingPlayers.Count(p => p.Position == PlayerPosition.GK);
    }

    public int NumberTransfers { get; }

    public decimal RemainingBudget { get; }

    public List<FplPlayer> OtherPlayers { get; }

    public List<FplPlayer> ExistingPlayers { get; }

    public FplOptions Options { get; }

    public List<FplPlayer> AllPlayers => ExistingPlayers.Union(OtherPlayers).ToList();

    public int Budget => ExistingPlayers.Sum(e => e.Cost) + (int)Math.Round(RemainingBudget * 10);
}