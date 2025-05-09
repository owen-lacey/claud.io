using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateTots;

public record CalculateTotsRequest : IRequest<SelectedTeam>;