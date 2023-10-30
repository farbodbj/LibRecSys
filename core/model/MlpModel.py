import BaseDeepLearningModel
from keras.layers import concatenate

class MlpModel(BaseDeepLearningModel):
    def __init__(self, user_count: int, item_count: int, latent_dim: int):
        super().__init__(user_count, item_count, latent_dim)
        
    
    def _buildMergeLayer(self):
        self.merge_layer = concatenate([self.user_embedding_layer, self.item_embedding_layer], axis = 1)
        return self.merge_layer
    
    def _buildPartialGraph(self):
        self.user_input_layer = self._buildInputLayer()
        self.item_input_layer = self._buildInputLayer()
        
        self.user_embedding_layer = self._buildEmbedding(nonNeg = False)(self.user_input_layer)
        self.item_embedding_layer = self._buildEmbedding(nonNeg = False)(self.item_input_layer)
        
        return self._buildMergeLayer()