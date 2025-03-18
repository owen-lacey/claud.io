using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Extensions;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateWildcard;

public class CalculateWildcardHandler(IFplRepository repository) : IRequestHandler<CalculateWildcardRequest, SelectedTeam>
{
    private readonly IFplRepository _repository = repository;

    public async Task<SelectedTeam> Handle(CalculateWildcardRequest request, CancellationToken cancellationToken)
    {
        var currentTeam = await _repository.GetSelectedTeamAsync(cancellationToken);
        var players = await _repository.GetPlayersAsync(cancellationToken);

        players.PopulateCostsFrom(currentTeam);
        var model = new PickFplTeamModel(players, FplOptions.RealWorld, currentTeam.Budget);
        var solver = new PickFplTeamSolver(model);

        var team = solver.Solve();

        return new SelectedTeam
        {
            StartingXi = team.StartingXi
                .OrderBy(p => p.Player.Position)
                .ThenByDescending(p => p.Player.XpNext)
                .ToList(),
            Bench = team.Bench
                .OrderBy(p => p.Player.Position)
                .ThenByDescending(p => p.Player.XpNext)
                .ToList(),
            FreeTransfers = currentTeam.FreeTransfers,
            Bank = currentTeam.Bank + currentTeam.SquadCost - team.SquadCost
        };
    }
}