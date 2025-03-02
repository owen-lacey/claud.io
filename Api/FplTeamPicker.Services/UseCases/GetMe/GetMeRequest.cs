using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetMe;

public record GetMeRequest : IRequest<User>;