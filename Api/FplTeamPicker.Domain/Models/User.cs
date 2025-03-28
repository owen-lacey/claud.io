namespace FplTeamPicker.Domain.Models;

public record User
{
    public string FirstName { get; init; }

    public string LastName { get; init; }

    public int Id { get; init; }
}