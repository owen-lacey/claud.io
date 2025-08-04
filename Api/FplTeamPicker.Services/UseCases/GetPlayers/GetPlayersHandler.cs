using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetPlayers;

public class GetPlayersHandler : IRequestHandler<GetPlayersRequest, List<Player>>
{
    private readonly IReferenceDataRepository _referenceDataRepository;

    public GetPlayersHandler(IReferenceDataRepository referenceDataRepository)
    {
        _referenceDataRepository = referenceDataRepository;
    }

    public async Task<List<Player>> Handle(GetPlayersRequest request, CancellationToken cancellationToken)
    {
        var players = await _referenceDataRepository.GetPlayersAsync(cancellationToken);

        return players
            .OrderByDescending(p => p.Xp).ToList();
    }
}