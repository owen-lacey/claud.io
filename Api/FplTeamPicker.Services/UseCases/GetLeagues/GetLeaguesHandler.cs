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
        return leagues;
    }
}