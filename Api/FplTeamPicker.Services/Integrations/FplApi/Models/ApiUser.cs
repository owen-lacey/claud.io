namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiUser
{
    public int Entry { get; set; }

    public string FirstName { get; set; } = null!;

    public string LastName { get; set; } = null!;
}