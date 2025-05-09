namespace FplTeamPicker.Domain.Models;

public record SelectedTeam
{
    public ICollection<SelectedPlayer> StartingXi { get; set; } = new List<SelectedPlayer>();

    public int SquadCost => StartingXi.Sum(p => p.SellingPrice);

    public decimal Score => StartingXi.Sum(p => p.Player.SeasonPoints);
}