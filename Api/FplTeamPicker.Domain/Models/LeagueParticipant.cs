namespace FplTeamPicker.Domain.Models;

public class LeagueParticipant
{
    public int UserId { get; set; }

    public string PlayerName { get; set; } = null!;

    public string TeamName { get; set; } = null!;

    public int Position { get; set; }
    
    public int Total { get; set; }
}