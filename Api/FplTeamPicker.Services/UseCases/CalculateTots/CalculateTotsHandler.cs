using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using FplTeamPicker.Services.Optimisation.UseCases.Tots;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateTots;

public class CalculateTotsHandler(IFplRepository repository) : IRequestHandler<CalculateTotsRequest, SelectedTeam>
{
    private readonly IFplRepository _repository = repository;

    public async Task<SelectedTeam> Handle(CalculateTotsRequest request, CancellationToken cancellationToken)
    {
        var players = await _repository.GetPlayersAsync(cancellationToken);

        var solver = new TotsSolver(new TotsModelInput
        {
            Players = players,
            Options = FplOptions.RealWorldXi,
            Budget = 1000
        });

        var result = solver.Solve();

        return new SelectedTeam
        {
            StartingXi = result.StartingXi
        };
    }
}