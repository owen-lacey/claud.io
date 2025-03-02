namespace FplTeamPicker.Domain.Models;

public record Transfers
{
    public List<SelectedPlayer> PlayersOut { get; set; }

    public List<SelectedPlayer> PlayersIn { get; set; }

    public ICollection<SelectedPlayer> StartingXi { get; set; } = [];

    public ICollection<SelectedPlayer> Bench { get; set; } = [];

    public int FreeTransfers { get; set; }

    public int Bank { get; set; }

    public int SquadCost => StartingXi.Sum(p => p.SellingPrice) + Bench.Sum(p => p.SellingPrice);

    public int Budget => Bank + SquadCost;

    public decimal PredictedPoints => StartingXi.Sum(p => p.Player.XpNext);
}