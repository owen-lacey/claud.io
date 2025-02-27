using System.Diagnostics.CodeAnalysis;
using Google.OrTools.Sat;

namespace FplTeamPicker.Optimisation.Models;

/// <summary>
/// Position-specific variables for player selections.
/// Only one of these variables will ever be set to true.
/// </summary>
public class FplPlayerSelectionVar
{
    [SetsRequiredMembers]
    public FplPlayerSelectionVar(CpModel model, FplPlayer player)
    {
        SquadSelected = model.NewBoolVar($"{player}_squad_selected");
        TeamSelected = model.NewBoolVar($"{player}_team_selected");
        Position = player.Position;
        Team = player.Team;
        Cost = player.Cost;
        PredictedPoints = player.PredictedPoints;
    }

    public IntVar SquadSelected { get; init; }

    public IntVar TeamSelected { get; init; }

    public PlayerPosition Position { get; init; }

    public required string Team { get; init; }

    public decimal Cost { get; init; }

    public decimal PredictedPoints { get; init; }
}