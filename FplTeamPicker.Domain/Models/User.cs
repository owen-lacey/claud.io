namespace FplTeamPicker.Domain.Models;

public record User
{
    public required string FirstName { get; init; }

    public required string LastName { get; init; }

    public required int Id { get; init; }
}