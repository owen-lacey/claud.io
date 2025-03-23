namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiEntryPicks
{
    public List<ApiTeamPick> Picks { get; set; } = [];
}