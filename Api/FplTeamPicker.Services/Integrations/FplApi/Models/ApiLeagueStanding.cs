namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public class ApiLeagueStanding
{
    public bool HasNext { get; set; }

    public List<ApiLeaguePosition> Results { get; set; }
}