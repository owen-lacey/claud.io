using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Integrations.FplApi.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetPlayers;

public class GetPlayersHandler : IRequestHandler<GetPlayersRequest, List<Player>>
{
    private readonly IFplRepository _fplRepository;

    public GetPlayersHandler(IFplRepository fplRepository)
    {
        _fplRepository = fplRepository;
    }

    public async Task<List<Player>> Handle(GetPlayersRequest request, CancellationToken cancellationToken)
    {
        var players = await _fplRepository.GetPlayersAsync(cancellationToken);

        return players
            .OrderByDescending(p => p.XpNext).ToList();
    }
}