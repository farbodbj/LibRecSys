import abc
import tensorflow as tf
from keras.layers import Embedding, Input, Flatten
from keras.constraints import NonNeg

class BaseDeepLearningModel():
    def __init__(self, user_count: int, item_count: int, latent_dim: int):
        self.user_count = user_count
        self.item_count = item_count
        self.latent_dim = latent_dim
        
        self.user_input_layer = None
        self.item_input_layer = None
        self.user_embedding_layer = None
        self.item_embedding_layer = None
        self.merge_layer = None
        
    def _buildInputLayer(self):
        return Input(shape = [1], dtype = tf.int32)
    
    def _buildEmbedding(self, nonNeg: bool = True, **kwargs):
        raw_embedding = Embedding(self.user_count, self.latent_dim, input_dim = 1, **kwargs)
        if nonNeg: 
            nonNegConst = NonNeg()(raw_embedding)
            return Flatten()(nonNegConst)
        
        return Flatten()(raw_embedding)
    
    @abc.abstractmethod 
    def _buildMergeLayer(self): raise NotImplementedError