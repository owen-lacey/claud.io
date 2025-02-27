using System.Globalization;
using System.Text;
using CsvHelper;
using FplTeamPicker.Api.IoC;
using FplTeamPicker.Api.Providers;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Optimisation;
using FplTeamPicker.Optimisation.Models;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddEndpointsApiExplorer()
    .AddSwaggerGen()
    .AddAntiforgery()
    .AddHttpContextAccessor()
    .AddFplApi()
    .AddScoped<IFplUserProvider, FplUserProvider>()
    .AddMemoryCache();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

List<string> existingPlayerNames =
[
    "Dean Henderson",
    "Bart Verbruggen",
    "Michael Keane",
    "Lucas Digne",
    "Jacob Greaves",
    "John Stones",
    "Rico Lewis",
    "Antoine Semenyo",
    "Emile Smith Rowe",
    "Mohamed Salah",
    "Bryan Mbeumo",
    "Georginio Rutter",
    "Nicolas Jackson",
    "Norberto Bercique Gomes Betuncal",
    "Chris Wood"
];

app.MapPost("/predictions", (IFormFile file) =>
    {
        using var stream = file.OpenReadStream();
        using var reader = new StreamReader(stream!);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        var players = csv.GetRecords<FplPlayer>()
            .Where(p => p.IsAvailable)
            .ToList();
        var existingPlayers = players.Where(p => existingPlayerNames.Contains(p.Name)).ToList();
        var allPlayers = players.ToList();

        return new
        {
            MyPlayers = existingPlayers.OrderByDescending(a => a.PredictedPoints).ToList(),
            BestPlayers = allPlayers.OrderByDescending(a => a.PredictedPoints).Take(15).ToList()
        };
    })
    .DisableAntiforgery();

app.MapPost("/wildcard", (IFormFile file, decimal budget) =>
    {
        using var stream = file.OpenReadStream();
        using var reader = new StreamReader(stream!);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        var players = csv.GetRecords<FplPlayer>()
            .Where(p => p.IsAvailable)
            .ToList();
        var options = FplOptions.RealWorld;
        var budgetMillions = (int)Math.Round(budget * 10);
        var model = new PickFplTeamModel(players, options, budgetMillions);
        var solver = new PickFplTeamSolver(model);

        var solution = solver.Solve();
        var sb = new StringBuilder().AppendLine("Player,Team,Position,Cost,PredictedPoints");
        solution.Squad.ForEach(p => sb.AppendLine($"{p.Name},{p.Team},{p.Position},{p.Cost},{p.PredictedPoints}"));
        return sb.ToString();
    })
    .DisableAntiforgery();

app.MapPost("/transfers", (IFormFile file, int numberTransfers, decimal remainingBudget, string? removePlayer) =>
    {
        using var stream = file.OpenReadStream();
        using var reader = new StreamReader(stream!);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        var players = csv.GetRecords<FplPlayer>()
            .Where(p => p.IsAvailable)
            .ToList();
        var existingPlayers = players.Where(p => existingPlayerNames.Contains(p.Name)).ToList();
        if (existingPlayers.Count < 15)
        {
            throw new Exception("Bad player names. Expected 15 players. Players found:" + string.Join(", ", existingPlayers.Select(p => p.Name)));
        }
        var otherPlayers = players.Except(existingPlayers).ToList();
        var options = FplOptions.RealWorld;
        if (removePlayer != null)
        {
            options.GetRidOf = existingPlayers.Single(p => p.Name == removePlayer);
        }
        var model = new FplTeamTransfersRequest(existingPlayers, otherPlayers, options, numberTransfers, remainingBudget);
        var solver = new FplTeamTransfersSolver(model);

        return solver.Solve();
    })
    .DisableAntiforgery();

app.MapGet("/me", async ([FromServices]IFplRepository repository, CancellationToken cancellationToken) =>
    {
        var result = await repository.GetUserDetails(cancellationToken);

        return Results.Ok(result);
    });

app.MapGet("/team", async ([FromServices]IFplRepository repository, CancellationToken cancellationToken) =>
    {
        var result = await repository.GetTeam(cancellationToken);

        return Results.Ok(result);
    });

app.Run();