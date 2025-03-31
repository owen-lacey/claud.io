namespace FplTeamPicker.Domain.Models;

public class Team
{
    public required string ShortName { get; set; }

    public required string Name { get; set; }

    public int Code { get; set; }

    public int Id { get; set; }
}