namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public class ApiEntry
{
    public required ApiLeagues Leagues { get; set; }

    public int SummaryOverallPoints { get; set; }
}