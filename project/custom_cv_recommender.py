import pandas as pd
import numpy as np
from typing import Dict, List, Any
from surprise import Reader, Dataset
from surprise.model_selection import KFold
from itertools import product
from tqdm import tqdm
from utils import get_top_k
from metrics import hit_rate, mean_reciprocal_rank


class CustomGridSearchCV:    
    def __init__(self, 
                 algorithm_class,
                 param_grid: Dict[str, List[Any]],
                 measures: List[str] = ['hit_rate', 'mrr'],
                 rating_relevance_threshold: float = 3.0,
                 k_cutoff: int = 10,
                 cv: int = 3,
                 random_state: int = 42):
        """
        Initialize the custom cross-validator.
        
        Args:
            algorithm_class: Surprise algorithm class (e.g., SVD, KNNWithMeans)
            param_grid: Dictionary of hyperparameters to search
            measures: List of metrics to evaluate ['hit_rate', 'mrr', 'map', 'precision']
            k_cutoff: Number of recommendations for top-K metrics
            cv: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.algorithm_class = algorithm_class
        self.param_grid = param_grid
        self.measures = measures
        self.k_cutoff = k_cutoff
        self.cv = cv
        self.random_state = random_state
        self.rating_relevance_threshold = rating_relevance_threshold
        
        self.best_params_ = {}
        self.best_score_ = {}
        
        self.metric_functions = {
            'hit_rate': hit_rate,
            'mrr': mean_reciprocal_rank
        }
    
    def fit(self, df_train: pd.DataFrame):
        df_train = df_train.sort_values('time')

        total_size = len(df_train)
        fold_size = total_size // self.cv
        fold_data = []
        
        for i in range(1, self.cv): # Start loop from 1
            # Train data is everything from the beginning up to the start of the test period
            train_end = i * fold_size
            train_data = df_train.iloc[0:train_end]

            # Test data is the current chunk
            test_end = (i + 1) * fold_size if i < self.cv - 1 else total_size
            test_data = df_train.iloc[train_end:test_end]

            # Handle the "cold start" problem for this fold
            # Ensure all users and items in test_data have been seen in train_data
            train_users = set(train_data['user_id'].unique())
            train_items = set(train_data['item_id'].unique())
            test_data = test_data[test_data['user_id'].isin(train_users)]
            test_data = test_data[test_data['item_id'].isin(train_items)]

            if len(test_data) == 0:
                continue # Skip fold if no valid test data

            # Convert to Surprise format...
            reader = Reader(rating_scale=(df_train['rating'].min(), df_train['rating'].max()))
            trainset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
            testset = [(row['user_id'], row['item_id'], row['rating']) 
                    for _, row in test_data.iterrows()]

            fold_data.append((trainset, testset))
        
                # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = [dict(zip(param_names, combo)) 
                            for combo in product(*param_values)]
        
        for measure in self.measures:
            self.best_score_[measure] = -np.inf
            self.best_params_[measure] = None
        
        for current_params in tqdm(param_combinations):
            cv_scores = {measure: [] for measure in self.measures}
            
            for trainset, testset in fold_data:
                algo = self.algorithm_class(**current_params)
                algo.fit(trainset)
                
                anti_testset = trainset.build_anti_testset()
                all_predictions = algo.test(anti_testset)
                
                test_df = pd.DataFrame([(uid, iid, r_ui) for uid, iid, r_ui in testset], 
                                     columns=['user_id', 'item_id', 'rating'])
                
                recommendations_per_user = get_top_k(all_predictions, self.k_cutoff)
                fold_scores = self._calculate_metrics(recommendations_per_user, test_df)
                
                for measure in self.measures:
                    cv_scores[measure].append(fold_scores[measure])
            
            avg_scores = {measure: np.mean(cv_scores[measure]) for measure in self.measures}
            
            for measure in self.measures:                
                if avg_scores[measure] > self.best_score_[measure]:
                    self.best_score_[measure] = avg_scores[measure]
                    self.best_params_[measure] = current_params.copy()
            
    def _calculate_metrics(self, recommendations_per_user: Dict[str, List[str]], test_df: pd.DataFrame) -> Dict[str, float]:
        scores = {}
        
        for measure in self.measures:
            if measure in self.metric_functions:
                scores[measure] = self.metric_functions[measure](
                    recommendations_per_user, test_df, self.k_cutoff, self.rating_relevance_threshold
                )
            else:
                raise ValueError(f"Unknown measure: {measure}")
        
        return scores
    
    def get_best_estimator(self, measure: str = None):
        if measure is None:
            measure = self.measures[0]
        
        if measure not in self.best_params_:
            raise ValueError(f"Measure '{measure}' not found. Available: {list(self.best_params_.keys())}")
        
        return self.algorithm_class(**self.best_params_[measure])
