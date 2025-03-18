namespace FplTeamPicker.Domain.Models;

public record SelectedTeam
{
    public ICollection<SelectedPlayer> StartingXi { get; set; } = [];

    public ICollection<SelectedPlayer> Bench { get; set; } = [];

    public int FreeTransfers { get; set; }

    public int Bank { get; set; }

    public int SquadCost => StartingXi.Sum(p => p.SellingPrice) + Bench.Sum(p => p.SellingPrice);

    public int Budget => Bank + SquadCost;

    public decimal PredictedPoints => StartingXi.Sum(p => p.Player.XpNext);
}

// FROM ENTRY endpoint, get all leagues with a type of "x"
// Then get the standings using /leagues-classic/{leagueId}/standings
// Then, look up the team of competitors using the `entry` of the results