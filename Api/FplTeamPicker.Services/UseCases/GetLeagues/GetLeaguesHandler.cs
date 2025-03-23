using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetLeagues;

public class GetLeaguesHandler : IRequestHandler<GetLeaguesRequest, List<League>>
{
    private readonly IFplRepository _fplRepository;

    public GetLeaguesHandler(IFplRepository fplRepository)
    {
        _fplRepository = fplRepository;
    }

    public async Task<List<League>> Handle(GetLeaguesRequest request, CancellationToken cancellationToken)
    {
        var leagues = await _fplRepository.GetLeaguesAsync(cancellationToken);

        var currentGameweek = await _fplRepository.GetCurrentGameweekAsync(cancellationToken);

        foreach (var participant in leagues.SelectMany(l => l.Participants))
        {
            var team = await _fplRepository.GetPreviousSelectedTeamAsync(participant.UserId, currentGameweek,
                cancellationToken);
            participant.StartingXi = team.StartingXi.Select(p => p.Player.Id).ToList();
            participant.Bench = team.Bench.Select(p => p.Player.Id).ToList();
        }

        return leagues;
    }
}