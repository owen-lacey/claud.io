using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetCurrentTeam;

public class GetCurrentTeamHandler : IRequestHandler<GetCurrentTeamRequest, SelectedSquad>
{
    private readonly IUserRepository _userRepository;
    private readonly IReferenceDataRepository _repository;

    public GetCurrentTeamHandler(IReferenceDataRepository repository, IUserRepository userRepository)
    {
        _repository = repository;
        _userRepository = userRepository;
    }

    public async Task<SelectedSquad> Handle(GetCurrentTeamRequest request, CancellationToken cancellationToken)
    {
        var gameweek = await _repository.GetCurrentGameweekAsync(cancellationToken);
        var selectedTeam = await _userRepository.GetSelectedTeamAsync(request.UserId, gameweek, cancellationToken);
        return selectedTeam;
    }
}