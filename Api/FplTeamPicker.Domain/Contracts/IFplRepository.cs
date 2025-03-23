using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IFplRepository
{
    Task<User> GetUserDetailsAsync(CancellationToken cancellationToken);

    Task<SelectedTeam> GetCurrentSelectedTeamAsync(CancellationToken cancellationToken);
    
    Task<SelectedTeam> GetPreviousSelectedTeamAsync(int userId, int gameweek, CancellationToken cancellationToken);
    
    Task<List<League>> GetLeaguesAsync(CancellationToken cancellationToken);

    Task<List<Player>> GetPlayersAsync(CancellationToken cancellationToken);
    
    Task<List<Team>> GetTeamsAsync(CancellationToken cancellationToken);
    
    Task<int> GetCurrentGameweekAsync(CancellationToken cancellationToken);
}