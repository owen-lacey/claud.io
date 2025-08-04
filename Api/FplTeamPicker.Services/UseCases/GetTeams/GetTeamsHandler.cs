using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetTeams;

public class GetTeamsHandler : IRequestHandler<GetTeamsRequest, List<Team>>
{
    private readonly IReferenceDataRepository _referenceDataRepository;

    public GetTeamsHandler(IReferenceDataRepository referenceDataRepository)
    {
        _referenceDataRepository = referenceDataRepository;
    }

    public async Task<List<Team>> Handle(GetTeamsRequest request, CancellationToken cancellationToken)
    {
        var teams = await _referenceDataRepository.GetTeamsAsync(cancellationToken);
        
        return teams;
    }
}