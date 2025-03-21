using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetTeams;

public record GetTeamsRequest : IRequest<List<Team>>;