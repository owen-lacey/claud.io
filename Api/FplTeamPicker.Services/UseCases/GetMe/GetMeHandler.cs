using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetMe;

public class GetMeHandler(IFplRepository repository) : IRequestHandler<GetMeRequest, User>
{
    private readonly IFplRepository _repository = repository;

    public Task<User> Handle(GetMeRequest request, CancellationToken cancellationToken)
    {
        return _repository.GetUserDetailsAsync(cancellationToken);
    }
}