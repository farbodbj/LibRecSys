from core.model.BaseClasses import BasePredictor
import numpy as np
import pandas as pd

class BaseRecommender():
    def __init__(self, model: BasePredictor, user_cols: list[str], item_cols: list[str]):
        self.model = model
        self.user_cols = user_cols
        self.item_cols = item_cols
        self.predictions = dict()
    
    
    def getTopForUser(self, user_id, count: int = 10): 
        self.predictions[user_id] = self.model.predictBatch(pd.Series(user_id, index = range(len(self.item_cols))), self.item_cols).flatten()
        topInds = np.argpartition(self.predictions, count)[-count:]
        
        return [[user_id, self.item_cols.iloc[idx], self.predictions[idx]] for idx in topInds]
    
    
    
        
        