using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateWildcard;

public record CalculateWildcardRequest : IRequest<Team>;