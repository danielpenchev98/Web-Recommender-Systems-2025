import math
from collections import defaultdict
import numpy as np

# Use Reciprocal Rank Fusion with k=60
def merge_recommendation_lists_using_rrf(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10, rrf_constant=60):
    hybrid_system_top_k_recommendations_per_user = {}
    
    users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    
    for user in users:
        reciprocal_rank_fusion_score_per_item = {}
        
        if user in recommendation_list_per_user_1.keys():
            recommendation_list = recommendation_list_per_user_1[user]
            for i, item in enumerate(recommendation_list):
                if item not in reciprocal_rank_fusion_score_per_item:
                    reciprocal_rank_fusion_score_per_item[item] = 1 / (rrf_constant + (i+1))
                else:
                    reciprocal_rank_fusion_score_per_item[item] += 1 / (rrf_constant + (i+1))
        
        if user in recommendation_list_per_user_2.keys():
            recommendation_list = recommendation_list_per_user_2[user]
            for i, item in enumerate(recommendation_list):
                if item not in reciprocal_rank_fusion_score_per_item:
                    reciprocal_rank_fusion_score_per_item[item] = 1 / (rrf_constant + (i+1))
                else:
                    reciprocal_rank_fusion_score_per_item[item] += 1 / (rrf_constant + (i+1))

        hybrid_system_top_k_recommendations_per_user[user] = [
            item_id for item_id, _ in sorted(reciprocal_rank_fusion_score_per_item.items(), key=lambda item: item[1], reverse=True)
        ][:cutoff]
        
    return hybrid_system_top_k_recommendations_per_user


def merge_recommendation_lists_using_round_robin(recommendation_list_per_user_1, recommendation_list_per_user_2, cutoff=10): 
    hybrid_system_top_k_recommendations_per_user = {}
    users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    
    for user in users:
        item_included = {}
        recommendation_list_1 = recommendation_list_per_user_1[user]
        recommendation_list_2 = recommendation_list_per_user_2[user]
        
        hybrid_recommendation_list = []
        iter_list_1, iter_list_2 = 0, 0

        while iter_list_1 < len(recommendation_list_1) and iter_list_2 < len(recommendation_list_2):
            item_id = recommendation_list_1[iter_list_1]
            if item_id not in item_included:
                hybrid_recommendation_list.append(item_id) 
                item_included[item_id] = {}
            iter_list_1+=1
            
            item_id = recommendation_list_2[iter_list_2]
            if item_id not in item_included:
                hybrid_recommendation_list.append(item_id)
                item_included[item_id] = {}
            iter_list_2+=1
            
        while iter_list_1 < len(recommendation_list_1):
            item_id = recommendation_list_1[iter_list_1]
            if item_id not in item_included:
                hybrid_recommendation_list.append(item_id) 
                item_included[item_id] = {}
            iter_list_1+=1
        
        while iter_list_2 <  len(recommendation_list_2):
            item_id = recommendation_list_2[iter_list_2]
            if item_id not in item_included:
                hybrid_recommendation_list.append(item_id)
                item_included[item_id] = {}
            iter_list_2+=1
        
        hybrid_system_top_k_recommendations_per_user[user] = hybrid_recommendation_list[:cutoff]
        
    return hybrid_system_top_k_recommendations_per_user

def combine_sum(recommendation_list_per_user_1,  recommendation_list_per_user_2, cutoff=10, model1_weight = 0.5, model2_weight=0.5):
    # for every user each entry in the list contains tuple (item, est_rating/similarity)
    
    hybrid_system_top_k_recommendations_per_user = {}
    users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    for user in users:
        # compute mean and std per list
        scores_model_1 = np.array([s for _, s in recommendation_list_per_user_1[user]])
        mean_1, std_1 = scores_model_1.mean(), scores_model_1.std() if scores_model_1.std() > 10**(-4) else 1.0

        scores_model_2 = np.array([s for _, s in recommendation_list_per_user_2[user]])
        mean_2, std_2 = scores_model_2.mean(), scores_model_2.std() if scores_model_2.std() > 10**(-4) else 1.0

        item_hybrid_score = defaultdict(float)
        for item, score in recommendation_list_per_user_1[user]:
            item_hybrid_score[item] += model1_weight * (score - mean_1)/std_1
        
        for item, score in recommendation_list_per_user_2[user]:
            item_hybrid_score[item] += model2_weight * (score - mean_2)/std_2
        
        hybrid_system_top_k_recommendations_per_user[user] = list(map(lambda x: x[0], sorted(item_hybrid_score.items(), key=lambda x: x[1], reverse=True)))[:cutoff]
    
    return hybrid_system_top_k_recommendations_per_user
        
def combine_max(recommendation_list_per_user_1,  recommendation_list_per_user_2, cutoff=10):
    # for every user each entry in the list contains tuple (item, est_rating/similarity)
    
    hybrid_system_top_k_recommendations_per_user = {}
    users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    for user in users:
        # compute mean and std per list
        scores_model_1 = np.array([s for _, s in recommendation_list_per_user_1[user]])
        mean_1, std_1 = scores_model_1.mean(), scores_model_1.std() if scores_model_1.std() > 10**(-4) else 1.0

        scores_model_2 = np.array([s for _, s in recommendation_list_per_user_2[user]])
        mean_2, std_2 = scores_model_2.mean(), scores_model_2.std() if scores_model_2.std() > 10**(-4) else 1.0

        item_hybrid_score = defaultdict(float)
        for item, score in recommendation_list_per_user_1[user]:
            item_hybrid_score[item] = (score - mean_1)/std_1
        
        for item, score in recommendation_list_per_user_2[user]:
            item_hybrid_score[item] = max(item_hybrid_score[item], (score - mean_2)/std_2)
        
        hybrid_system_top_k_recommendations_per_user[user] = list(map(lambda x: x[0], sorted(item_hybrid_score.items(), key=lambda x: x[1], reverse=True)))[:cutoff]
    
    return hybrid_system_top_k_recommendations_per_user

def combine_min(recommendation_list_per_user_1,  recommendation_list_per_user_2, cutoff=10, model1_weight = 0.5, model2_weight=0.5):
    # for every user each entry in the list contains tuple (item, est_rating/similarity)
    
    hybrid_system_top_k_recommendations_per_user = {}
    users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    for user in users:
        # compute mean and std per list
        scores_model_1 = np.array([s for _, s in recommendation_list_per_user_1[user]])
        mean_1, std_1 = scores_model_1.mean(), scores_model_1.std() if scores_model_1.std() > 10**(-4) else 1.0

        scores_model_2 = np.array([s for _, s in recommendation_list_per_user_2[user]])
        mean_2, std_2 = scores_model_2.mean(), scores_model_2.std() if scores_model_2.std() > 10**(-4) else 1.0

        item_hybrid_score = defaultdict(float)
        for item, score in recommendation_list_per_user_1[user]:
            item_hybrid_score[item] = (score - mean_1)/std_1
        
        for item, score in recommendation_list_per_user_2[user]:
            item_hybrid_score[item] = min(item_hybrid_score[item], (score - mean_2)/std_2)
        
        hybrid_system_top_k_recommendations_per_user[user] = list(map(lambda x: x[0], sorted(item_hybrid_score.items(), key=lambda x: x[1], reverse=True)))[:cutoff]
    
    return hybrid_system_top_k_recommendations_per_user
        
def combine_mnz(recommendation_list_per_user_1,  recommendation_list_per_user_2, cutoff=10, model1_weight = 0.5, model2_weight=0.5):
    # for every user each entry in the list contains tuple (item, est_rating/similarity)
    
    hybrid_system_top_k_recommendations_per_user = {}
    users = set(recommendation_list_per_user_1.keys()) | set(recommendation_list_per_user_2.keys())
    
    item_occurrences = defaultdict(int)
    for user in users:
        # compute mean and std per list
        scores_model_1 = np.array([s for _, s in recommendation_list_per_user_1[user]])
        mean_1, std_1 = scores_model_1.mean(), scores_model_1.std() if scores_model_1.std() > 10**(-4) else 1.0

        scores_model_2 = np.array([s for _, s in recommendation_list_per_user_2[user]])
        mean_2, std_2 = scores_model_2.mean(), scores_model_2.std() if scores_model_2.std() > 10**(-4) else 1.0

        item_hybrid_score = defaultdict(float)
        for item, score in recommendation_list_per_user_1[user]:
            item_hybrid_score[item] += model1_weight * (score - mean_1)/std_1
            item_occurrences[item] += 1
            
        for item, score in recommendation_list_per_user_2[user]:
            item_hybrid_score[item] += model2_weight * (score - mean_2)/std_2
            item_occurrences[item] += 1
        
        hybrid_system_top_k_recommendations_per_user[user] = list(map(lambda x: x[0], sorted(item_hybrid_score.items(), key=lambda x: x[1] * item_occurrences[x[0]], reverse=True)))[:cutoff]
    
    return hybrid_system_top_k_recommendations_per_user