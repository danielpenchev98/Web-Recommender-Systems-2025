from utils import *
import numpy as np
from typing import Set

def precision_at_k(recommendations_per_user: Dict[str,List[str]], 
                   df_test: pd.DataFrame,
                   k: int) -> Dict[str, float]:
    """Compute precision at k for each user
    Args:
        recommendations_per_user: Dictionary of recommendations for each test user.
        df_test: Pandas DataFrame containing user-item ratings in the test split.
        k: cutoff for the metric measurement
    Returns:
        The P@k (averaged of the users)
    """

    precisions = {}
    
    top_k_recommended_items_relevance_per_user = { 
            user : convert_ratings_to_relevance(user, recommendations[:k], df_test)
            for user, recommendations in recommendations_per_user.items()
            if user in df_test.user_id.tolist()
        }

    # Divide by k even if the list is shorter - way of punishing the algorithm
    # If we don't punish the algorithm then the best strategy for the algorithm 
    # to game the metric is to output only a single recommendation that algo is certain that it is relevant
    precisions = [
        sum(top_k_recommended_items_relevance)/k 
        for top_k_recommended_items_relevance in top_k_recommended_items_relevance_per_user.values()
    ]
    
    return sum(precisions)/len(precisions) 


def mean_average_precision(recommendations_per_user:  Dict[str,List[str]], 
                           df_test: pd.DataFrame,
                           k: int) -> float:
    """Compute the mean average precision 
    Args:
        recommendations_per_user: Dictionary of recommendations for each test user.
        df_test: Pandas DataFrame containing user-item ratings in the test split.
        k: cutoff for the metric measurement
    Returns:
        The MAP@k (averaged of the users) 
    """

    average_precision_per_users = []
    # order all items in descending order with respect to predicted rating

    
    # map each recommendation to either 1 (if relevant) or 0 (not relevant)
    top_k_recommended_items_relevance_per_user = { 
            user : convert_ratings_to_relevance(user, recommendations[:k], df_test)
            for user, recommendations in recommendations_per_user.items()
            if user in df_test.user_id.tolist()
        }
    
    for user, top_k_recommended_items_relevance in top_k_recommended_items_relevance_per_user.items(): 
        average_precision = 0
        
        num_hits = 0
        for i in range(len(top_k_recommended_items_relevance)):
            if top_k_recommended_items_relevance[i] == 0.0:
                continue
            num_hits += 1
            #average_precision += sum(num_hits)/(i+1)
            #assert sum(top_k_recommended_items_relevance[:i+1])/(i+1) == num_hits/(i+1)
            average_precision += num_hits/(i+1) #sum(top_k_recommended_items_relevance[:i+1])/(i+1)

        num_relevant_items_for_user_in_test_dataset = get_num_relevant_items(user, df_test)
        
        # if no relevant items for that user, then average_precision is 0 anyway
        if num_relevant_items_for_user_in_test_dataset > 0:
            # Dividing using min(k, num_relevant_items_for_user_in_test_dataset) instead of 
            average_precision /= min(k, num_relevant_items_for_user_in_test_dataset)
        #else:
            #average_precision = 0.0

        average_precision_per_users.append(average_precision)

    # averaging over all users
    mapk = np.mean(average_precision_per_users)
    return mapk

def mean_reciprocal_rank(recommendations_per_user: Dict[str,List[str]], 
                         df_test: pd.DataFrame,
                         k: int) -> float:
    """Compute the mean reciprocal rank 
    Args:
        recommendations_per_user: Dictionary of recommendations for each test user.
        df_test: Pandas DataFrame containing user-item ratings in the test split.
        k: cutoff for the metric measurement
    Returns:
        The MRR@k (averaged of the users) 
    """    
    reciprocal_rank = []

    top_k_recommended_items_relevance_per_user = { 
            user : convert_ratings_to_relevance(user, recommendations[:k], df_test)
            for user, recommendations in recommendations_per_user.items()
            if user in df_test.user_id.tolist()
        }
    
    for top_k_recommended_items_relevance in top_k_recommended_items_relevance_per_user.values():
        # some lists might not contain any relevant items. What we need to find is the first relevant item  
        reciprocal_rank.append(0.0)
        for i, relevance in enumerate(top_k_recommended_items_relevance):
            if relevance == 1:
                reciprocal_rank[-1] = (1.0/(i+1))
                break
    
    mean_rr =  np.mean(reciprocal_rank) if len(reciprocal_rank) > 0 else 0
    return mean_rr

def hit_rate(recommendations_per_user: Dict[str, List[str]],
             df_test: pd.DataFrame,
             k: int) -> float:
    """Compute the hit rate
    Args:
        recommendations_per_user: Dictionary of recommendations for each test user.
        df_test: Pandas DataFrame containing user-item ratings in the test split.
        k: cutoff for the metric measurement
    Returns:
        The Hit@k (averaged of the users) 
    """

    hits = 0.0
    
    top_k_recommended_items_relevance_per_user = { 
            user : convert_ratings_to_relevance(user, recommendations[:k], df_test)
            for user, recommendations in recommendations_per_user.items()
            if user in df_test.user_id.tolist()
        }

    for top_k_recommended_items_relevance in top_k_recommended_items_relevance_per_user.values():
        if sum(top_k_recommended_items_relevance) > 0:
            hits += 1.0
    
    num_users = len(top_k_recommended_items_relevance_per_user.keys())
    return hits/max(1,num_users)


def coverage(recommendation_ordered_items_per_user: Dict[str,List[str]],
             catalog_items: Set[str],
             k: int) -> float:
    """Compute the catalog coverage
    Args:
        recommendations_per_user: Dictionary of recommendations for each test user.
        catalog_items: Set of all items that the recommender system knows about
        k: cutoff for the metric measurement
    Returns:
        The Coverage@k (averaged of the users) 
    """
    
    all_recommended_items = set(
        item 
        for recommended_items in recommendation_ordered_items_per_user.values()
        for item in recommended_items[:k] # is this the correct way
    )
    
    all_recommended_items_from_the_catalog = all_recommended_items & catalog_items
    return len(all_recommended_items_from_the_catalog) * 1.0 / len(catalog_items)