import torch

def growth_metric(values)
    """
    Calculate the growth metric based on only likes

    Args:
        values (torch.tensor): Tensor containing likes to evaluate for the metric with batch dimension

    Returns:
        torch.tensor: Tensor containing the metric with batch dimension
    """

    # Compute difference of values
    values = torch.diff(values, dim=1)
    # Sum across batches dimensions
    values = torch.sum(values, dim=1)
    return values


def loss(model, dataloader, loss_metric=torch.nn.functional.mse_loss(), metric=growth_metric):
    """
    Calculate the loss of the model

    Args:
        model (torch.nn): Model
        dataloader (DataLoaders): Dataloader
        loss_metricc: Metric used to evaluate the preiction
        metric: The metric used to create an evaluation of the output data

    Returns:
        double: The average loss
    """
    with torch.no_grad():
        correct = 0
        loss = 0
        for data in dataloader:
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = model(x) 
            truth = growth_metric(y)

            correct += (pred == truth).int().sum()
            
            loss += loss_metric(pred.squeeze(), y.squeeze())            

    return correct/len(dataloader.dataset), loss/len(dataloader.dataset)
