using FplTeamPicker.Domain.Models;

namespace FplTeamPicker.Domain.Contracts;

public interface IFplRepository
{
    Task<User> GetUserDetails(CancellationToken cancellationToken);

    Task<Team> GetTeam(CancellationToken cancellationToken);
}