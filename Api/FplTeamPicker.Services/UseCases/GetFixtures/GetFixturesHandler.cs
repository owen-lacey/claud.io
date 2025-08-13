using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetFixtures;

public class GetFixturesHandler : IRequestHandler<GetFixturesRequest, List<Fixture>>
{
    private readonly IReferenceDataRepository _referenceDataRepository;

    public GetFixturesHandler(IReferenceDataRepository referenceDataRepository)
    {
        _referenceDataRepository = referenceDataRepository;
    }

    public async Task<List<Fixture>> Handle(GetFixturesRequest request, CancellationToken cancellationToken)
    {
        var fixtures = await _referenceDataRepository.GetFixturesAsync(cancellationToken);
        var currentGameweek = await _referenceDataRepository.GetCurrentGameweekAsync(cancellationToken);

        // Get the next 6 gameweeks of fixtures for all teams
        var upcomingFixtures = fixtures
            .Where(f => !f.Finished && f.Gameweek >= currentGameweek && f.Gameweek <= currentGameweek + 5)
            .OrderBy(f => f.Gameweek)
            .ThenBy(f => f.KickoffTime)
            .ToList();

        return upcomingFixtures;
    }
}
