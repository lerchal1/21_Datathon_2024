import torch
def growth_metric(values):
    """
    Calculate the growth metric based on

    Args:
        values (torch.tensor): Tensor containing likes to evaluate for the metric with batch dimension

    Returns:
        torch.tensor: Tensor containing the metric with batch dimension
    """
    likes_w = 0.6
    comments_w = 0.2
    follow_gain_w = 0.2


    # Compute difference of values
    # Keep followers,pictures,videos,comments,likes,Posts ratio
    values = (follow_gain_w*values[:, 0] + comments_w*values[:, 3] + likes_w*values[:, 4])/values[:, 5]

    # Sum across batches dimensions
    value = torch.mean(values)
    return value

