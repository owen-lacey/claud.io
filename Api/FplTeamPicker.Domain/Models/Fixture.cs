namespace FplTeamPicker.Domain.Models;

public class Fixture
{
    public int Id { get; set; }
    public int Gameweek { get; set; }
    public string Season { get; set; } = string.Empty;
    public int TeamHome { get; set; }
    public int TeamAway { get; set; }
    public int TeamHomeDifficulty { get; set; }
    public int TeamAwayDifficulty { get; set; }
    public DateTime KickoffTime { get; set; }
    public bool Finished { get; set; }
    public int? TeamHomeScore { get; set; }
    public int? TeamAwayScore { get; set; }
}
