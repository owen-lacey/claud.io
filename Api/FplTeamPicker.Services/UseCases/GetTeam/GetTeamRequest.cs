using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetTeam;

public record GetTeamRequest : IRequest<Team>
{
}