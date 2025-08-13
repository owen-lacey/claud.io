using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetFixtures;

public record GetFixturesRequest : IRequest<List<Fixture>>;
