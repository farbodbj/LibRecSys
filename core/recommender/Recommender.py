from core.model.BaseClasses import BasePredictor
import numpy as np


class BaseRecommender():
    def __init__(self, model: BasePredictor, user_cols: list[str], item_cols: list[str]):
        self.model = model
        self.user_cols = user_cols
        self.item_cols = item_cols
    
    
    def getTop(self, count: int = 10): 
        preds = self.model.predictBatch(self.user_cols, self.item_cols).flatten()
        topInds = np.argpartition(preds, count)[-count:]
        
        return [(self.user_cols.iloc[idx], self.item_cols.iloc[idx], preds[idx]) for idx in topInds]
    
    
    
        
        