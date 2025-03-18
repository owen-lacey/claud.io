namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public class ApiLeague
{
    public int Id { get; set; }

    public string Name { get; set; } = null!;

    public string LeagueType { get; set; } = null!;

    public int EntryRank { get; set; }
}