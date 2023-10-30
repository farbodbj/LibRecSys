import BaseMatrixFactorizationModel
import tensorflow as tf
from keras.layers import Embedding, Input, Multiply, Dense, Layer
from keras.constraints import NonNeg

class GmfModel(BaseMatrixFactorizationModel):
    def __init__(self, user_count: int, item_count: int, latent_dim: int):
        super().__init__(user_count, item_count, latent_dim)
        
        self.user_embedding = None
        self.item_embedding = None
        
    def _buildGmfInputLayer(self):
        return Input(shape = [1], dtype = tf.int32)
    
    def _buildGmfEmbedding(self, nonNeg: bool = True, **kwargs):
        raw_embedding = Embedding(self.user_count, self.latent_dim, input_dim = 1, **kwargs)
        if nonNeg: 
            return NonNeg()(raw_embedding)
        
        return raw_embedding
    
    def _buildMergeLayer(self):
        return Multiply()([self.user_embedding, self.item_embedding])
    
    def _buildPartialGraph(self):
        self.user_input = self._buildGmfInputLayer()
        self.item_input = self._buildGmfInputLayer()
        
        self.user_embedding = self._buildGmfEmbedding(nonNeg = True)
        self.item_embedding = self._buildGmfEmbedding(nonNeg = True)
        
        return self._buildMergeLayer()
        