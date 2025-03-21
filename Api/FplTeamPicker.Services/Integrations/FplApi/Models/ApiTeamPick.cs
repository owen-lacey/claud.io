using System.Text.Json.Serialization;

namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiTeamPick
{
    [JsonPropertyName("element")]
    public int Id { get; set; }

    [JsonPropertyName("element_type")]
    public int Position { get; set; }

    public bool IsCaptain { get; set; }

    public bool IsViceCaptain { get; set; }

    public int SellingPrice { get; set; }

    public int PurchasePrice { get; set; }

    [JsonPropertyName("position")]
    public int SquadNumber { get; set; }
}