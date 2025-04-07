
from typing import List
import numpy as np
import pandas as pd

class TopPopularNew:
    
    def __init__(self):
        self.trained = False
        self.recommenations = None
        self.scores = None
        self.train_interactions = None
    
    def fit(self, df: pd.DataFrame, alpha: float = 6):

        counts = df.groupby('item_id').size()
        counts_tr = df.groupby('item_id')['target'].sum()

        #добавляем параметризацию, чтобы избежать рекомендаций по релевантным, но непопулярным айтемам
        popularity = counts_tr/(counts + alpha)

        self.recommenations = popularity.sort_values(ascending=False).index.to_list()
        self.scores = popularity
        self.trained = True
        self.train_interactions = df.groupby('user_id')['item_id'].agg(set)

    def predict(self, df_train: pd.DataFrame, df_test: pd.DataFrame, topn: int = 6) -> List[List[int]]:
        assert self.trained

        base_recs = self.recommenations
        train_interactions = self.train_interactions

        def get_recommendations(user):
            if user not in train_interactions.index:
                return base_recs[:topn]
            else:
                user_train_interactions = train_interactions.loc[user]
                return [item for item in base_recs if item not in user_train_interactions][:topn]

        return df_test['user_id'].apply(lambda x: get_recommendations(x)).tolist()
    
    
    def get_score(self, df_train: pd.DataFrame, df_test: pd.DataFrame)  -> float:
        
        assert self.trained

        train_interactions = self.train_interactions
        scores = self.scores

        def get_scores(user, item):
            if user not in train_interactions.index or item not in scores.index:
                return scores.mean()
            else:
                user_train_interactions = train_interactions.loc[user]
                score = scores[item]
            if item in user_train_interactions:
                return 0
            else:
                return score
        
        return df_test.apply(lambda row: get_scores(row['user_id'], row['item_id']), axis=1).tolist()

        

    

    