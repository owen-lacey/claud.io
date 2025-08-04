using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Extensions;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using FplTeamPicker.Services.Optimisation.UseCases.Transfers;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateTransfers;

public class CalculateTransfersHandler
    : IRequestHandler<CalculateTransfersRequest, Transfers>
{
    public CalculateTransfersHandler(IReferenceDataRepository repository, IUserRepository userRepository)
    {
        _repository = repository;
        _userRepository = userRepository;
    }
    private readonly IReferenceDataRepository _repository;
    private readonly IUserRepository _userRepository;

    public async Task<Transfers> Handle(CalculateTransfersRequest request, CancellationToken cancellationToken)
    {
        var currentTeam = await _userRepository.GetMyTeamAsync(cancellationToken);
        var players = await _repository.GetPlayersAsync(cancellationToken);

        players.PopulateCostsFrom(currentTeam);

        var existingPlayers = currentTeam.SelectedSquad.StartingXi
            .Concat(currentTeam.SelectedSquad.Bench)
            .OrderBy(r => r.Player.Id)
            .ToList();
        var otherPlayers = players.Where(p => existingPlayers.All(ep => ep.Player.Id != p.Id)).ToList();
        var solverRequest = new TransfersModelInput(
            existingPlayers.Select(p => p.Player).ToList(),
            otherPlayers,
            FplOptions.RealWorld,
            currentTeam.FreeTransfers,
            currentTeam.Bank);
        var solver = new TransfersSolver(solverRequest);
        var transfers = solver.Solve();

        var playersIn = transfers.StartingXi
            .Concat(transfers.Bench)
            .Where(p => existingPlayers.All(ep => ep.Player.Id != p.Player.Id))
            .ToList();

        var playersOut = existingPlayers
            .Where(p => transfers.Bench.Concat(transfers.StartingXi).All(ep => ep.Player.Id != p.Player.Id))
            .ToList();

        return new Transfers
        {
            PlayersIn = playersIn,
            PlayersOut = playersOut,
            StartingXi = transfers.StartingXi
                .OrderBy(p => p.Player.Position)
                .ThenByDescending(p => p.Player.Xp)
                .ToList(),
            Bench = transfers.Bench
                .OrderBy(p => p.Player.Position)
                .ThenByDescending(p => p.Player.Xp)
                .ToList(),
            FreeTransfers = currentTeam.FreeTransfers - playersIn.Count,
            Bank = currentTeam.Bank + currentTeam.SelectedSquad.SquadCost - transfers.SquadCost
        };
    }
}