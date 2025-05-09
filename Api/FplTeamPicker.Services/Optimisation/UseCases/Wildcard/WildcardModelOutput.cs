using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Services.Optimisation.UseCases.Wildcard;

public class WildcardModelOutput
{
    public List<SelectedPlayer> Bench { get; set; } = [];

    public List<SelectedPlayer> StartingXi { get; set; } = [];

    public int SquadCost => Bench.Sum(p => p.Player.Cost) + StartingXi.Sum(p => p.Player.Cost);
}