using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetPlayers;

public record GetPlayersRequest(
    int? PlayerId = null,
    int? TeamId = null,
    string? Position = null,
    int? Limit = null
) : IRequest<List<Player>>;
