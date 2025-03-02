using FplTeamPicker.Api.IoC;
using FplTeamPicker.Api.Providers;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Services.UseCases.CalculateTransfers;
using FplTeamPicker.Services.UseCases.CalculateWildcard;
using FplTeamPicker.Services.UseCases.GetMe;
using FplTeamPicker.Services.UseCases.GetPlayers;
using FplTeamPicker.Services.UseCases.GetTeam;
using MediatR;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddEndpointsApiExplorer()
    .AddSwaggerGen()
    .AddAntiforgery()
    .AddHttpContextAccessor()
    .AddFplApi()
    .AddScoped<IFplUserProvider, FplUserProvider>()
    .AddMemoryCache()
    .AddMediatR(r => r.RegisterServicesFromAssemblyContaining<GetMeRequest>());

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapGet("/transfers", async ([FromServices]IMediator mediator, CancellationToken cancellationToken) =>
    {
        var result = await mediator.Send(new CalculateTransfersRequest(), cancellationToken);

        return Results.Ok(result);
    });

app.MapGet("/me", async ([FromServices]IMediator mediator, CancellationToken cancellationToken) =>
    {
        var result = await mediator.Send(new GetMeRequest(), cancellationToken);

        return Results.Ok(result);
    });

app.MapGet("/team", async ([FromServices]IMediator mediator, CancellationToken cancellationToken) =>
    {
        var result = await mediator.Send(new GetTeamRequest(), cancellationToken);

        return Results.Ok(result);
    });

app.MapGet("/players", async ([FromServices]IMediator mediator, CancellationToken cancellationToken) =>
    {
        var result = await mediator.Send(new GetPlayersRequest(), cancellationToken);

        return Results.Ok(result);
    });

app.MapGet("/wildcard", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var team = await mediator.Send(new CalculateWildcardRequest(), cancellationToken);
    return Results.Ok(team);
});

app.Run();