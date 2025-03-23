namespace FplTeamPicker.Domain.Models;

public class LeagueParticipant
{
    public int UserId { get; set; }

    public string PlayerName { get; set; } = null!;

    public string TeamName { get; set; } = null!;

    public int Position { get; set; }

    public List<int> StartingXi { get; set; }

    public List<int> Bench { get; set; }
}