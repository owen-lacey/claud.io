using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetCurrentTeam;

public class GetCurrentTeamHandler : IRequestHandler<GetCurrentTeamRequest, SelectedTeam>
{
    private readonly IFplRepository _repository;

    public GetCurrentTeamHandler(IFplRepository repository)
    {
        _repository = repository;
    }

    public async Task<SelectedTeam> Handle(GetCurrentTeamRequest request, CancellationToken cancellationToken)
    {
        var gameweek = await _repository.GetCurrentGameweekAsync(cancellationToken);
        var selectedTeam = await _repository.GetSelectedTeamAsync(request.UserId, gameweek, cancellationToken);
        return selectedTeam;
    }
}