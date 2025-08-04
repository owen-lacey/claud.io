using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IUserRepository
{
    Task<User> GetUserDetailsAsync(CancellationToken cancellationToken);

    Task<MyTeam> GetMyTeamAsync(CancellationToken cancellationToken);
    
    Task<SelectedSquad> GetSelectedTeamAsync(int userId, int gameweek, CancellationToken cancellationToken);
    
    Task<List<League>> GetLeaguesAsync(CancellationToken cancellationToken);
}