
namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiDataDump
{
    public required List<ApiPlayerDetails> Elements { get; set; }
    
    public required List<ApiTeamDetails> Teams { get; set; }
    
    public required List<ApiEvent> Events { get; set; }
}