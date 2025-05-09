using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetCurrentTeam;

public record GetCurrentTeamRequest(int UserId) : IRequest<SelectedSquad>;