using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetTeam;

public class GetTeamHandler(IFplRepository repository) : IRequestHandler<GetTeamRequest, Team>
{
    private readonly IFplRepository _repository = repository;

    public Task<Team> Handle(GetTeamRequest request, CancellationToken cancellationToken)
    {
        var team = _repository.GetTeamAsync(cancellationToken);
        return team;
    }
}