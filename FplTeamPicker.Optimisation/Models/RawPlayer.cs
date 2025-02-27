using CsvHelper.Configuration.Attributes;

namespace FplTeamPicker.Optimisation.Models;

public class RawPlayer
{
    [Name("name")]
    public string Name { get; set; } = null!;

    [Name("team")]
    public string Team { get; set; } = null!;

    /// <summary>
    /// Gets or sets raw cost in millions.
    /// </summary>
    [Name("cost")]
    public decimal CostMillions { get; set; }

    /// <summary>
    /// Gets the cost in an OR-Tools friendly format (i.e no decimals).
    /// Assumes cost  never has > 1dp specificity.
    /// </summary>
    public int Cost => (int)(CostMillions * 10);

    [Name("pos")]
    public PlayerPosition Position { get; set; }

    [Name("sel")]
    public int PercentSelectedBy { get; set; }

    // expected_assists_per_90
    //
    // expected_goal_involvements_per_90
    //
    // expected_goals_per_90
    //
    // expected_goals_conceded_per_90
    //
    // goals_conceded_per_90
    //
    //     points_per_game
    // saves_per_90
    //     selected_by_percent
    // starts_per_90

    public override string ToString()
    {
        return $"{Name}-{Team}-{Position}".Replace(" ", "_");
    }
}