import torch
def growth_metric(values):
    """
    Calculate the growth metric for a brand on fixed window as an average of interactions/(nr_posts + 1) in that window
    This should give an idea of the growth of the brand in that period compared to *its* previous

    Args:
        values (torch.tensor): Tensor containing features to evaluate for the metric with batch dimension
        Order of inputs : followers,pictures,videos,comments,likes,Posts


    Returns:
        torch.tensor: Tensor containing the metric with batch dimension
    """
    comments_weight = 0.2
    likes_weight = 0.6
    followers_growth_weight = 0.2

    # Compute metric of growth: interactions/(nr_posts + 1) (+1 is to avoid dividing by 0) per week in the given window
    val = (followers_growth_weight*values[:, 0] + comments_weight*values[:, 3] + likes_weight*values[:, 4])/(values[:, 5] + 1)
    # Average across window to get mean growth in the period
    value = torch.mean(values[:, 0])
    return value
