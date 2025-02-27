using CsvHelper.Configuration.Attributes;

namespace FplTeamPicker.Optimisation.Models;

public class FplPlayer
{
    [Name("PlayerName")]
    public string Name { get; set; } = null!;

    [Name("Team")]
    public string Team { get; set; } = null!;

    /// <summary>
    /// Gets or sets cost in 10's of millions
    /// </summary>
    [Name("Value")]
    public int Cost { get; set; }

    [Name("Position")]
    public PlayerPosition Position { get; set; }

    [Name("PredictedPoints")]
    public decimal PredictedPoints { get; set; }

    [Name("ChanceOfPlayingNextRound")]
    public string ChanceOfPlayingNextRound { get; set; }

    public bool IsAvailable => true;

    public override string ToString()
    {
        return $"{Name}-{Team}-{Position}".Replace(" ", "_");
    }
}