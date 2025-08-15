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

        var filtered = players.AsQueryable();

        if (request.PlayerId.HasValue)
            filtered = filtered.Where(p => p.Id == request.PlayerId.Value);
        if (request.TeamId.HasValue)
            filtered = filtered.Where(p => p.Team == request.TeamId.Value);
        if (!string.IsNullOrEmpty(request.Position))
            filtered = filtered.Where(p => p.Position.ToString().Equals(request.Position, StringComparison.OrdinalIgnoreCase));

        var result = filtered
            .OrderByDescending(p => p.Xp)
            .ToList();

        if (request.Limit.HasValue)
            result = result.Take(request.Limit.Value).ToList();

        return result;
    }
}