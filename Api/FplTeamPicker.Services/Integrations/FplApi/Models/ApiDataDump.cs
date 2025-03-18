using System.Text.Json.Serialization;

namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiDataDump
{
    [JsonPropertyName("elements")]
    public List<ApiPlayerDetails> Players { get; set; }
    
    public List<ApiTeamDetails> Teams { get; set; }
}