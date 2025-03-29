using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.Models;

public class FplTeamSelection
{
    public List<SelectedPlayer> Bench { get; set; } = new List<SelectedPlayer>();

    public List<SelectedPlayer> StartingXi { get; set; } = new List<SelectedPlayer>();

    public int SquadCost => Bench.Sum(p => p.Player.Cost) + StartingXi.Sum(p => p.Player.Cost);
}