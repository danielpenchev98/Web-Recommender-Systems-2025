from surprise.prediction_algorithms.predictions import Prediction
from typing import Dict, List
import pandas as pd

def get_top_k(predictions: List[Prediction], 
              k: int,
              include_est_rating: bool=False) -> Dict[str, List]:
    """Compute the top-K recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        k(int): The number of recommendation to output for each user. If  k is -1, then no predictions are discarded.
        include_est_rating(bool): flag on whether to include the estimated rating or not in the output
    Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    
    pred_df = pd.DataFrame(predictions)

    # Select top-k predictions per user
    filtered_pred_df = pred_df.sort_values(["uid", "est"], ascending=[True, False]) \
            .groupby("uid", group_keys=True, as_index=False) \
            .head(k)

    # Convert to dictionary of lists
    if include_est_rating:
        return filtered_pred_df.groupby("uid").apply(lambda g: list(zip(g["iid"], g["est"]))).to_dict()    
    return filtered_pred_df.groupby("uid")['iid'].apply(list).to_dict()

def convert_ratings_to_relevance(user: str,
                                  recommendations: List[str],
                                  dataset: pd.DataFrame, rating_relevance_threshold: float = 3.0) -> float:
    return [1  if ((dataset.user_id == user) & (dataset.item_id == iid) & (dataset.rating >= rating_relevance_threshold)).any() else 0 for iid in recommendations]

def get_num_relevant_items(user: str,
                            dataset: pd.DataFrame, rating_relevance_threshold: float = 3.0) -> float:
    return dataset[(dataset.user_id == user) & (dataset.rating >= rating_relevance_threshold)].shape[0]
