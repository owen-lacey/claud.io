using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.Models;

public class FplTransfersSelection
{
    public List<SelectedPlayer> Bench { get; set; } = [];

    public List<SelectedPlayer> StartingXi { get; set; } = [];

    public int SquadCost => Bench.Sum(p => p.Player.Cost) + StartingXi.Sum(p => p.Player.Cost);
}