using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetPlayers;

public record GetPlayersRequest : IRequest<List<Player>>;