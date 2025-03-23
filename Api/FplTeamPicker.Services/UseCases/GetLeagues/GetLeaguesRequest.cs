using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetLeagues;

public record GetLeaguesRequest : IRequest<List<League>>;