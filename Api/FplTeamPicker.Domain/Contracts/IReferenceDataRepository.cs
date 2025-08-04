using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IReferenceDataRepository
{
    Task<List<Player>> GetPlayersAsync(CancellationToken cancellationToken);

    Task<List<Team>> GetTeamsAsync(CancellationToken cancellationToken);

    Task<int> GetCurrentGameweekAsync(CancellationToken cancellationToken);
}