namespace FplTeamPicker.Domain.Models;

public class League
{
    public int Id { get; set; }
    
    public string Name { get; set; } = null!;

    public int CurrentPosition { get; set; }

    public List<LeagueParticipant> Participants { get; set; } = new List<LeagueParticipant>();

    public int NumberOfPlayers => Participants.Count;
}