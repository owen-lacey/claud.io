namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiEvent
{
    public int Id { get; set; }
    
    public bool IsCurrent { get; set; }
}