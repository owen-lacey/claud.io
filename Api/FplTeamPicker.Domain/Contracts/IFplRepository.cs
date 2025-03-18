using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IFplRepository
{
    Task<User> GetUserDetailsAsync(CancellationToken cancellationToken);

    Task<SelectedTeam> GetSelectedTeamAsync(CancellationToken cancellationToken);

    Task<List<Player>> GetPlayersAsync(CancellationToken cancellationToken);
    
    Task<List<Team>> GetTeamsAsync(CancellationToken cancellationToken);
}