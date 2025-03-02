using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Extensions;
using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation;
using FplTeamPicker.Services.Optimisation.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateTransfers;

public class CalculateTransfersHandler(IFplRepository repository)
    : IRequestHandler<CalculateTransfersRequest, Transfers>
{
    private readonly IFplRepository _repository = repository;

    public async Task<Transfers> Handle(CalculateTransfersRequest request, CancellationToken cancellationToken)
    {
        var currentTeam = await _repository.GetTeamAsync(cancellationToken);
        var players = await _repository.GetPlayersAsync(cancellationToken);

        players.PopulateCostsFrom(currentTeam);

        var existingPlayers = currentTeam.StartingXi
            .Concat(currentTeam.Bench)
            .OrderBy(r => r.Player.Id)
            .ToList();
        var otherPlayers = players.Where(p => existingPlayers.All(ep => ep.Player.Id != p.Id)).ToList();
        var solverRequest = new FplTeamTransfersRequest(
            existingPlayers.Select(p => p.Player).ToList(),
            otherPlayers,
            FplOptions.RealWorld,
            currentTeam.FreeTransfers,
            currentTeam.Bank);
        var solver = new FplTeamTransfersSolver(solverRequest);
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
                .ThenByDescending(p => p.Player.XpNext)
                .ToList(),
            Bench = transfers.Bench
                .OrderBy(p => p.Player.Position)
                .ThenByDescending(p => p.Player.XpNext)
                .ToList(),
            FreeTransfers = currentTeam.FreeTransfers,
            Bank = currentTeam.Bank + currentTeam.SquadCost - transfers.SquadCost
        };
    }
}