import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_embedding_item_representation(item_ids_and_item_text_representations, word_embeddings):
    embedding_size = word_embeddings.vectors.shape[1]

    item_embeddings = {}
    for item_id, text_representation in item_ids_and_item_text_representations:
        in_vocab_words = 0
        item_embedding = np.zeros((embedding_size,))
        for word in text_representation.split():
            if word not in word_embeddings:
                continue
            item_embedding += word_embeddings[word]
            in_vocab_words+=1
        
        item_embeddings[item_id] = (item_embedding / max(in_vocab_words,1))
    
    return pd.DataFrame.from_dict(item_embeddings, orient='index')

def calculate_user_profile(user, df_train, df_item_embeddings, list_items_with_valid_metadata):
    user_item_interactions_for_items_with_metadata = df_train[
        (df_train.user_id==user) & (df_train.item_id.isin(list_items_with_valid_metadata))
    ].sort_values('item_id')
    
    items_with_metadata_rated_by_user = user_item_interactions_for_items_with_metadata.item_id.tolist()
    item_embeddings = df_item_embeddings.loc[items_with_metadata_rated_by_user]

    rating_values_for_items = user_item_interactions_for_items_with_metadata.rating.to_numpy()
    return np.array(np.mean(item_embeddings * (rating_values_for_items[:, np.newaxis] - rating_values_for_items.mean()), axis=0))

def find_unobserved_items_for_user(user, df_train, all_valid_items):
    items_rated_by_the_user = set(df_train[df_train.user_id == user].item_id.tolist())
    unobserved_items = [item for item in all_valid_items if item not in items_rated_by_the_user]
    return set(unobserved_items)

def generated_recommendations_for_user(user_profile, unrated_user_items, df_item_embeddings, cutoff):
    sorted_unrated_user_items = sorted(list(unrated_user_items))
    unrated_items_item_embeddings = df_item_embeddings.loc[sorted_unrated_user_items]

    result = cosine_similarity(user_profile.reshape(1, -1), unrated_items_item_embeddings)

    user_recommendation_predictions_and_sim = sorted(list(zip(sorted_unrated_user_items, result[0])), key=lambda x: x[1], reverse=True)[:cutoff]

    user_recommendation_predictions = [item for item, _ in user_recommendation_predictions_and_sim]
    return user_recommendation_predictions, user_recommendation_predictions_and_sim