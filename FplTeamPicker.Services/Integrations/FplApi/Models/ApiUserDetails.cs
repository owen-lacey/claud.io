using System.Text.Json.Serialization;

namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiUserDetails
{
    [JsonPropertyName("player")]
    public ApiUser User { get; set; } = null!;
}