namespace FplTeamPicker.Domain.Models;

public record SelectedSquad
{
    public ICollection<SelectedPlayer> StartingXi { get; set; } = new List<SelectedPlayer>();

    public ICollection<SelectedPlayer> Bench { get; set; } = new List<SelectedPlayer>();

    public int SquadCost => StartingXi.Sum(p => p.SellingPrice) + Bench.Sum(p => p.SellingPrice);

    public decimal PredictedPoints => StartingXi.Sum(p => p.Player.XpNext);

    public decimal BenchBoostPredictedPoints => PredictedPoints + Bench.Sum(p => p.Player.XpNext);
}