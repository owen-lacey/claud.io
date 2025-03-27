namespace FplTeamPicker.Domain.Models;

public record SelectedTeam
{
    public ICollection<SelectedPlayer> StartingXi { get; set; } = [];

    public ICollection<SelectedPlayer> Bench { get; set; } = [];

    public int SquadCost => StartingXi.Sum(p => p.SellingPrice) + Bench.Sum(p => p.SellingPrice);

    public decimal PredictedPoints => StartingXi.Sum(p => p.Player.XpNext);

    public decimal BenchBoostPredictedPoints => PredictedPoints + Bench.Sum(p => p.Player.XpNext);
}