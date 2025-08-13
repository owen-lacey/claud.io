using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateWildcard;

public record CalculateWildcardRequest : IRequest<MyTeam>
{
    /// <summary>
    /// Gets or sets the list of player IDs that are locked in the team.
    /// </summary>
    public List<int> Locks { get; set; } = [];
}