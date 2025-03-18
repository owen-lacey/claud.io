using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetTeam;

public class GetSelectedTeamHandler(IFplRepository repository) : IRequestHandler<GetSelectedTeamRequest, SelectedTeam>
{
    private readonly IFplRepository _repository = repository;

    public Task<SelectedTeam> Handle(GetSelectedTeamRequest request, CancellationToken cancellationToken)
    {
        var team = _repository.GetSelectedTeamAsync(cancellationToken);
        return team;
    }
}