import math
from collections import defaultdict
import numpy as np
from itertools import zip_longest


# Use Reciprocal Rank Fusion with k=60
def merge_recommendation_lists_using_rrf(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10, rrf_constant=60):
    all_users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    hybrid_recs = {}
    
    for user in all_users:
        scores = defaultdict(float)

        for rank, item in enumerate(recommendation_list_per_user_1.get(user, []), 1):
            scores[item] += 1 / (rrf_constant + rank)
            
        for rank, item in enumerate(recommendation_list_per_user_2.get(user, []), 1):
            scores[item] += 1 / (rrf_constant + rank)
        
        sorted_items = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
        hybrid_recs[user] = sorted_items[:cutoff]
        
    return hybrid_recs

def merge_recommendation_lists_using_round_robin(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10):
    all_users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    hybrid_recs = {}

    for user in all_users:
        recs1 = recommendation_list_per_user_1.get(user, [])
        recs2 = recommendation_list_per_user_2.get(user, [])
        
        interleaved = []
        for item1, item2 in zip_longest(recs1, recs2):
            if item1 is not None:
                interleaved.append(item1)
            if item2 is not None:
                interleaved.append(item2)
        
        # De-duplicate while preserving order
        seen = set()
        final_list = [item for item in interleaved if not (item in seen or seen.add(item))]
        
        hybrid_recs[user] = final_list[:cutoff]
        
    return hybrid_recs

def combine_sum(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10, model1_weight=0.5, model2_weight=0.5):
    if model1_weight == 0.0:
        return {user: rec_list[:cutoff] for user, rec_list in recommendation_list_per_user_2.items()}
    elif model2_weight == 0.0:
        return {user: rec_list[:cutoff] for user, rec_list in recommendation_list_per_user_1.items()}
    
    
    mean1, std1 = _calculate_global_stats(recommendation_list_per_user_1)
    mean2, std2 = _calculate_global_stats(recommendation_list_per_user_2)

    all_users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    
    hybrid_recs = {}
    for user in all_users:
        final_scores = defaultdict(float)
        
        # 2. Normalize each score using the pre-calculated global stats.
        for item, score in recommendation_list_per_user_1.get(user, []):
            final_scores[item] += model1_weight * (score - mean1) / std1
            
        for item, score in recommendation_list_per_user_2.get(user, []):
            final_scores[item] += model2_weight * (score - mean2) / std2
            
        sorted_items = sorted(final_scores.keys(),key=lambda item: final_scores[item], reverse=True)
        hybrid_recs[user] = sorted_items[:cutoff]
        
    return hybrid_recs

def combine_max(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10):
    mean1, std1 = _calculate_global_stats(recommendation_list_per_user_1)
    mean2, std2 = _calculate_global_stats(recommendation_list_per_user_2)
    
    all_users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    
    hybrid_recs = {}
    for user in all_users:
        scores = defaultdict(lambda: -np.inf)
        
        for item, score in recommendation_list_per_user_1.get(user, []):
            scores[item] = max(scores[item], (score - mean1) / std1)
        for item, score in recommendation_list_per_user_2.get(user, []):
            scores[item] = max(scores[item], (score - mean2) / std2)
            
        sorted_items = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
        hybrid_recs[user] = sorted_items[:cutoff]
        
    return hybrid_recs

def combine_min(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10):
    mean1, std1 = _calculate_global_stats(recommendation_list_per_user_1)
    mean2, std2 = _calculate_global_stats(recommendation_list_per_user_2)
    
    all_users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    
    hybrid_recs = {}
    for user in all_users:
        scores = defaultdict(lambda: np.inf)
        
        for item, score in recommendation_list_per_user_1.get(user, []):
            scores[item] = min(scores[item], (score - mean1) / std1)
        for item, score in recommendation_list_per_user_2.get(user, []):
            scores[item] = min(scores[item], (score - mean2) / std2)
            
        sorted_items = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
        hybrid_recs[user] = sorted_items[:cutoff]
        
    return hybrid_recs

def combine_mnz(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10, model1_weight=0.5, model2_weight=0.5):
    if model1_weight == 0.0:
        return {user: rec_list[:cutoff] for user, rec_list in recommendation_list_per_user_2.items()}
    elif model2_weight == 0.0:
        return {user: rec_list[:cutoff] for user, rec_list in recommendation_list_per_user_1.items()}
    
    mean1, std1 = _calculate_global_stats(recommendation_list_per_user_1)
    mean2, std2 = _calculate_global_stats(recommendation_list_per_user_2)
    
    all_users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())

    hybrid_recs = {}
    for user in all_users:
        accum_weighted_scores = defaultdict(float)
        occurrences = defaultdict(int)
        
        for item, score in recommendation_list_per_user_1.get(user, []):
            accum_weighted_scores[item] += model1_weight * (score - mean1) / std1
            occurrences[item] += 1
        for item, score in recommendation_list_per_user_2.get(user, []):
            accum_weighted_scores[item] += model2_weight * (score - mean2) / std2
            occurrences[item] += 1
        
        sorted_items = sorted(accum_weighted_scores.keys(), 
                              key=lambda item: accum_weighted_scores[item] * occurrences[item], 
                              reverse=True)
        hybrid_recs[user] = sorted_items[:cutoff]
        
    return hybrid_recs

def switching_strategy_cold_start_rule(df_train, min_interactions):
   user_interaction_counts = df_train['user_id'].value_counts()
   return lambda user: user_interaction_counts.get(user, 0) < min_interactions


# Rule 2: User Rating Behavior
def switching_strategy_user_rating_behaviour_rule(df_train, rating_threshold = 3.0, conservative_threshold: float = 0.8,
                           liberal_threshold: float = 0.3):
        
        user_stats = {}
        for user in df_train['user_id'].unique():
            user_data = df_train[df_train['user_id'] == user]
            
            user_stats[user] = {
                'rating_std': user_data['rating'].std(),
                'high_ratings_ratio': (user_data['rating'] >= rating_threshold).mean(),
            }
        
        def rule(user):            
            if user not in user_stats:
                return True
            
            stats = user_stats[user]
            
            # Conservative raters -> Content-based
            if (stats['high_ratings_ratio'] > conservative_threshold and 
                stats['rating_std'] < 1.0):
                return True
            
            # Liberal raters -> KNN
            if stats['high_ratings_ratio'] < liberal_threshold:
                return False
            
            return True
        
        return rule

def merge_recommendation_lists_switching_strategy(
    primary_recommendation_list_per_user: dict, 
    backup_recommendation_list_per_user: dict, 
    rule, 
    cutoff=10
):
    hybrid_recs_per_user = {}
    
    all_users = set(primary_recommendation_list_per_user.keys()) | set(backup_recommendation_list_per_user.keys())
    
    for user in all_users:
        use_backup_model = rule(user)
        final_list = backup_recommendation_list_per_user.get(user, []) if use_backup_model else primary_recommendation_list_per_user.get(user, [])
        hybrid_recs_per_user[user] = final_list[:cutoff]
        
    return hybrid_recs_per_user

def _calculate_global_stats(rec_list_per_user):
    all_scores = np.array([score for recs in rec_list_per_user.values() for _, score in recs])
    std = all_scores.std()
    return all_scores.mean(), std if std > 1e-6 else 1.0
