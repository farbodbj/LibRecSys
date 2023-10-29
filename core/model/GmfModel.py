import BaseMatrixFactorizationModel
import tensorflow as tf
from keras.layers import Embedding, Input
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