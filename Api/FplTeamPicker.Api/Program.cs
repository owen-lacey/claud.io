using FplTeamPicker.Api.IoC;
using FplTeamPicker.Api.Providers;
using FplTeamPicker.Domain.Contracts;
using FplTeamPicker.Services.Integrations.FplApi.Constants;
using FplTeamPicker.Services.UseCases.CalculateTransfers;
using FplTeamPicker.Services.UseCases.CalculateWildcard;
using FplTeamPicker.Services.UseCases.GetCurrentTeam;
using FplTeamPicker.Services.UseCases.GetLeagues;
using FplTeamPicker.Services.UseCases.GetMe;
using FplTeamPicker.Services.UseCases.GetMyTeam;
using FplTeamPicker.Services.UseCases.GetPlayers;
using FplTeamPicker.Services.UseCases.GetTeams;
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
    .AddMediatR(r => r.RegisterServicesFromAssemblyContaining<GetMeRequest>())
    .AddCors(options => options.AddDefaultPolicy(corsBuilder =>
    {
        var uiUrl = builder.Configuration.GetValue<string>("EndpointConfig:UiUrl");
        corsBuilder.WithOrigins(uiUrl!)
            .WithMethods(HttpMethods.Get, HttpMethods.Post)
            .WithHeaders(FplApiConstants.HeaderName);
    }));

var app = builder.Build();

app
    .UseCors()
    .UseExceptionHandler(exceptionHandlerApp =>
    {
        exceptionHandlerApp.Run(Results.Problem().ExecuteAsync);
    });
;

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapPost("/transfers", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new CalculateTransfersRequest(), cancellationToken);

    return Results.Ok(result);
});

app.MapPost("/wildcard", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var team = await mediator.Send(new CalculateWildcardRequest(), cancellationToken);
    return Results.Ok(team);
});

app.MapGet("/my-details", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new GetMeRequest(), cancellationToken);

    return Results.Ok(result);
});

app.MapGet("/my-team", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new GetMyTeamRequest(), cancellationToken);

    return Results.Ok(result);
});

app.MapGet("/my-leagues", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new GetLeaguesRequest(), cancellationToken);
    
    return Results.Ok(result);
});

app.MapGet("/teams", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new GetTeamsRequest(), cancellationToken);

    return Results.Ok(result);
});

app.MapGet("/players", async ([FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new GetPlayersRequest(), cancellationToken);

    return Results.Ok(result);
});

app.MapGet("/users/{userId}/current-team", async ([FromRoute]int userId, [FromServices] IMediator mediator, CancellationToken cancellationToken) =>
{
    var result = await mediator.Send(new GetCurrentTeamRequest(userId), cancellationToken);

    return Results.Ok(result);
});

app.Run();