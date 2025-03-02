using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IFplRepository
{
    Task<User> GetUserDetailsAsync(CancellationToken cancellationToken);

    Task<Team> GetTeamAsync(CancellationToken cancellationToken);

    Task<List<Player>> GetPlayersAsync(CancellationToken cancellationToken);
}