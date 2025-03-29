namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiTeam
{
    public DateTimeOffset PicksLastUpdated { get; set; }

    public ICollection<ApiTeamPick> Picks { get; set; } = new List<ApiTeamPick>();

    public ApiTransfersDetails Transfers { get; set; } = null!;
}