using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IFplRepository
{
    Task<User> GetUserDetailsAsync(CancellationToken cancellationToken);

    Task<MyTeam> GetMyTeamAsync(CancellationToken cancellationToken);
    
    Task<SelectedTeam> GetSelectedTeamAsync(int userId, int gameweek, CancellationToken cancellationToken);
    
    Task<List<League>> GetLeaguesAsync(CancellationToken cancellationToken);

    Task<List<Player>> GetPlayersAsync(CancellationToken cancellationToken);
    
    Task<List<Team>> GetTeamsAsync(CancellationToken cancellationToken);
    
    Task<int> GetCurrentGameweekAsync(CancellationToken cancellationToken);
}