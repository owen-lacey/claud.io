namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiTransfersDetails
{
    public int Bank { get; set; }

    public int Value { get; set; }

    public int? Limit { get; set; }

    public int Made { get; set; }
}