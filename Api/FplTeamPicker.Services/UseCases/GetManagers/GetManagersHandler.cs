using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetManagers;

public class GetManagersHandler : IRequestHandler<GetManagersRequest, List<Manager>>
{
    private readonly IFplRepository _repository;

    public GetManagersHandler(IFplRepository repository)
    {
        _repository = repository;
    }

    public async Task<List<Manager>> Handle(GetManagersRequest request, CancellationToken cancellationToken)
    {
        var managers = await _repository.GetManagersAsync(cancellationToken);

        return managers
            .OrderByDescending(p => p.XpNext).ToList();
    }
}