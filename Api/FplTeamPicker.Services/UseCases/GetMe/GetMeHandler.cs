using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.GetMe;

public class GetMeHandler : IRequestHandler<GetMeRequest, User>
{
    private readonly IFplRepository _repository;

    public GetMeHandler(IFplRepository repository)
    {
        _repository = repository;
    }

    public Task<User> Handle(GetMeRequest request, CancellationToken cancellationToken)
    {
        return _repository.GetUserDetailsAsync(cancellationToken);
    }
}