from core.model.BaseMatrixFactorizationModel import BaseMatrixFactorizationModel
import tensorflow as tf
from keras.layers import Embedding, Input, Multiply, Flatten
from keras.constraints import NonNeg

class GmfModel(BaseMatrixFactorizationModel):
    def __init__(self, user_count: int, item_count: int, latent_dim: int):
        super().__init__(user_count, item_count, latent_dim)
        
        self.user_input = None
        self.item_input = None
        self.user_embedding = None
        self.item_embedding = None
        self.merge_layer = None
    
    def _buildGmfEmbedding(self, input, nonNeg: bool = True, **kwargs):
        raw_embedding = Embedding(input_dim = self.user_count, output_dim = self.latent_dim, input_length = 1, **kwargs)(input)
        if nonNeg: 
            nonNegConst = NonNeg()(raw_embedding)
            return Flatten()(nonNegConst)
        
        return Flatten()(raw_embedding)
    
    def _buildMergeLayer(self):
        self.merge_layer = Multiply()([self.user_embedding, self.item_embedding])
        return self.merge_layer
    
    def _buildPartialGraph(self):        
        self.user_embedding = self._buildGmfEmbedding(self.user_input, nonNeg = True)
        self.item_embedding = self._buildGmfEmbedding(self.item_input, nonNeg = True)
        
        return self._buildMergeLayer()
        