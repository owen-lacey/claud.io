using System.Text.Json.Serialization;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Integrations.FplApi.Extensions;

namespace FplTeamPicker.Services.Integrations.FplApi.Models;

public record ApiPlayerDetails
{
    public int Id { get; init; }

    [JsonPropertyName("first_name")] public string FirstName { get; init; } = null!;

    [JsonPropertyName("second_name")] public string SecondName { get; init; } = null!;

    [JsonPropertyName("ep_next")] public string? XpNext { get; set; }

    [JsonPropertyName("ep_this")] public string XpThis { get; set; } = null!;

    [JsonPropertyName("now_cost")] public int Cost { get; set; }

    [JsonPropertyName("element_type")] public ApiPosition Position { get; set; }

    public int Team { get; set; }

    public string SelectedByPercent { get; set; }

    public int TotalPoints { get; set; }

    public DateOnly? BirthDate { get; set; }

    public int YellowCards { get; set; }

    public int TransfersOut { get; set; }

    public int RedCards { get; set; }

    public Player ToPlayer()
    {
        return new Player
        {
            Position = Position.ToPosition(),
            SecondName = SecondName,
            FirstName = FirstName,
            XpNext = XpNext != null ? decimal.Parse(XpNext) : 0,
            XpThis = decimal.Parse(XpThis),
            Cost = Cost,
            Id = Id,
            Team = Team,
            SeasonPoints = TotalPoints,
            SelectedByPercent = decimal.Parse(SelectedByPercent),
            BirthDate = BirthDate,
            YellowCards = YellowCards,
            RedCards = RedCards,
            TransfersOut = TransfersOut,
        };
    }

    public Manager ToManager()
    {
        return new Manager
        {
            SecondName = SecondName,
            FirstName = FirstName,
            XpNext = decimal.Parse(XpNext),
            XpThis = decimal.Parse(XpThis),
            Cost = Cost,
            Id = Id,
            Team = Team
        };
    }
}