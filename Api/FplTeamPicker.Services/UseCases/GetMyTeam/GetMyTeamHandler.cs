using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetMyTeam;

public class GetMyTeamHandler : IRequestHandler<GetMyTeamRequest, MyTeam>
{
    private readonly IFplRepository _repository;

    public GetMyTeamHandler(IFplRepository repository)
    {
        _repository = repository;
    }

    public Task<MyTeam> Handle(GetMyTeamRequest request, CancellationToken cancellationToken)
    {
        var team = _repository.GetMyTeamAsync(cancellationToken);
        return team;
    }
}