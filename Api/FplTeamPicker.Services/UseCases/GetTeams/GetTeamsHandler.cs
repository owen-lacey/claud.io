using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetTeams;

public class GetTeamsHandler : IRequestHandler<GetTeamsRequest, List<Team>>
{
    private readonly IFplRepository _fplRepository;

    public GetTeamsHandler(IFplRepository fplRepository)
    {
        _fplRepository = fplRepository;
    }

    public async Task<List<Team>> Handle(GetTeamsRequest request, CancellationToken cancellationToken)
    {
        var teams = await _fplRepository.GetTeamsAsync(cancellationToken);
        
        return teams;
    }
}