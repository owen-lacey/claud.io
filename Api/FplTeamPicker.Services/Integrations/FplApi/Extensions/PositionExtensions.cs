using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Integrations.FplApi.Models;

namespace FplTeamPicker.Services.Integrations.FplApi.Extensions;

public static class PositionExtensions
{
    public static Position ToPosition(this ApiPosition position)
    {
        return position switch {
            ApiPosition.Goalkeeper => Position.Goalkeeper,
            ApiPosition.Defender => Position.Defender,
            ApiPosition.Midfielder => Position.Midfielder,
            ApiPosition.Forward => Position.Forward,
            _ => throw new ArgumentOutOfRangeException(nameof(position), position, null)
        };
    }
}