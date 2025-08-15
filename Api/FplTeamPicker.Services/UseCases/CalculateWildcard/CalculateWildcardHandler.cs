using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Extensions;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using FplTeamPicker.Services.Optimisation.UseCases.Wildcard;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateWildcard;

public class CalculateWildcardHandler : IRequestHandler<CalculateWildcardRequest, MyTeam>
{
    private readonly IReferenceDataRepository _repository;
    private readonly IUserRepository _userRepository;

    public CalculateWildcardHandler(IReferenceDataRepository repository, IUserRepository userRepository)
    {
        _repository = repository;
        _userRepository = userRepository;
    }

    public async Task<MyTeam> Handle(CalculateWildcardRequest request, CancellationToken cancellationToken)
    {
        var currentTeam = await _userRepository.GetMyTeamAsync(cancellationToken);
        var players = await _repository.GetPlayersAsync(cancellationToken);

        players.PopulateCostsFrom(currentTeam);
        var model = new WildcardModelInput(players, FplOptions.RealWorld, currentTeam.Budget - 50, request.Locks);
        var solver = new WildcardSolver(model);

        var team = solver.Solve();

        return new MyTeam
        {
            SelectedSquad = new SelectedSquad
            {
                StartingXi = team.StartingXi
                    .OrderBy(p => p.Player.Position)
                    .ThenByDescending(p => p.Player.Xp)
                    .ToList(),
                Bench = team.Bench
                    .OrderBy(p => p.Player.Position)
                    .ThenByDescending(p => p.Player.Xp)
                    .ToList()
            },
            FreeTransfers = currentTeam.FreeTransfers,
            Bank = currentTeam.Bank + currentTeam.SelectedSquad.SquadCost - team.SquadCost
        };
    }
}