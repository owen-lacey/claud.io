using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetMyTeam;

public record GetMyTeamRequest : IRequest<MyTeam>
{
}