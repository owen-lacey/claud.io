using FplTeamPicker.Domain.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.Models;

public class TeamSelectionVar
{
    public TeamSelectionVar(CpModel model, Player player)
    {
        Id = player.Id;
        Selected = model.NewBoolVar($"{player.Id}_team_selected");
        Position = player.Position;
        Team = player.Team;
        Cost = player.Cost;
    }

    public int Id { get; init; }

    public IntVar Selected { get; init; }

    public Position Position { get; init; }

    public int Team { get; init; }

    public decimal Cost { get; init; }
}