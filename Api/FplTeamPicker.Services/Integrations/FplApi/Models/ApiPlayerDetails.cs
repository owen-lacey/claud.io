using System.Text.Json.Serialization;

namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiPlayerDetails
{
    public int Id { get; init; }

    [JsonPropertyName("first_name")]
    public string FirstName { get; init; } = null!;

    [JsonPropertyName("second_name")]
    public string SecondName { get; init; } = null!;

    [JsonPropertyName("ep_next")]
    public string XpNext { get; set; } = null!;

    [JsonPropertyName("ep_this")]
    public string XpThis { get; set; } = null!;

    [JsonPropertyName("now_cost")]
    public int Cost { get; set; }
}