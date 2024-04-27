import torch
def growth_metric(values):
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
    value = torch.sum(values[:, 0])
    return value
