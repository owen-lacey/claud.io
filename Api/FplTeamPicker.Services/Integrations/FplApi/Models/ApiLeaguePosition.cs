namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public class ApiLeaguePosition
{
    public string PlayerName { get; set; } = null!;

    public string TeamName { get; set; } = null!;
    
    public int Entry { get; set; }

    public int Rank { get; set; }

    public int Total { get; set; }
}