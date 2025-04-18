using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetManagers;

public record GetManagersRequest : IRequest<List<Manager>>;