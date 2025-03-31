using System.Text.Json.Serialization;

namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiDataDump
{
    [JsonPropertyName("elements")]
    public required List<ApiPlayerDetails> Players { get; set; }
    
    public required List<ApiTeamDetails> Teams { get; set; }
    
    public required List<ApiEvent> Events { get; set; }
}