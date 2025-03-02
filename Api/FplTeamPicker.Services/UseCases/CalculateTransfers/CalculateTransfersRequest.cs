using FplTeamPicker.Domain.Models;
using MediatR;

namespace FplTeamPicker.Services.UseCases.CalculateTransfers;

public record CalculateTransfersRequest : IRequest<Transfers>
{
}