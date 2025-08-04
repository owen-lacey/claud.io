using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetLeagues;

public class GetLeaguesHandler : IRequestHandler<GetLeaguesRequest, List<League>>
{
    private readonly IUserRepository _userRepository;

    public GetLeaguesHandler(IUserRepository userRepository)
    {
        _userRepository = userRepository;
    }

    public async Task<List<League>> Handle(GetLeaguesRequest request, CancellationToken cancellationToken)
    {
        var leagues = await _userRepository.GetLeaguesAsync(cancellationToken);
        return leagues;
    }
}