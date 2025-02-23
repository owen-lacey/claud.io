using CsvHelper.Configuration.Attributes;

namespace FplTeamPicker.Models;

public record Fixture
{
    [Name("team_h")]
    public Team HomeTeam { get; set; }

    [Name("team_a")]
    public Team AwayTeam { get; set; }

    [Name("kickoff_time")]
    public DateTime KickOffTime { get; set; }

    [Name("team_h_difficulty")]
    public int HomeTeamDifficulty { get; set; }

    [Name("team_a_difficulty")]
    public int AwayTeamDifficulty { get; set; }
}