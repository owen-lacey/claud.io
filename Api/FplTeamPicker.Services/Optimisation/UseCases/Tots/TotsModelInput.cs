using FplTeamPicker.Domain.Models;
using FplTeamPicker.Services.Optimisation.Models;

namespace FplTeamPicker.Services.Optimisation.UseCases.Tots;

public record TotsModelInput
{
    public FplOptions Options { get; set; }
    
    public List<Player> Players { get; set; }
    
    public int Budget { get; set; }
}