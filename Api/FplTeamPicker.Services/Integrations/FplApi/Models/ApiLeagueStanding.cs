namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public class ApiLeagueStanding
{
    public bool HasNext { get; set; }

    public required List<ApiLeaguePosition> Results { get; set; }
}