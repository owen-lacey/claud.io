using System.Diagnostics.CodeAnalysis;
using FplTeamPicker.Domain.Models;
using Google.OrTools.Sat;

namespace FplTeamPicker.Services.Optimisation.Models;

/// <summary>
/// Position-specific variables for player selections.
/// Only one of these variables will ever be set to true.
/// </summary>
public class FplPlayerSelectionVar
{
    public FplPlayerSelectionVar(CpModel model, Player player)
    {
        Id = player.Id;
        SquadSelected = model.NewBoolVar($"{player}_squad_selected");
        TeamSelected = model.NewBoolVar($"{player}_team_selected");
        Position = player.Position;
        Team = player.Team;
        Cost = player.Cost;
        PredictedPoints = player.XpNext;
    }

    public int Id { get; init; }

    public IntVar SquadSelected { get; init; }

    public IntVar TeamSelected { get; init; }

    public Position Position { get; init; }

    public int Team { get; init; }

    public decimal Cost { get; init; }

    public decimal PredictedPoints { get; init; }
}