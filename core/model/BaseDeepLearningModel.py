import abc
import tensorflow as tf
from keras.layers import Embedding, Input, Flatten
from keras.constraints import NonNeg

class BaseDeepLearningModel():
    def __init__(self, user_count: int, item_count: int, latent_dim: int):
        self.user_count = user_count
        self.item_count = item_count
        self.latent_dim = latent_dim
        
        self.user_input = None
        self.item_input = None
        self.user_embedding_layer = None
        self.item_embedding_layer = None
        self.merge_layer = None
        
    
    def _buildEmbedding(self, input, nonNeg: bool = True, **kwargs):
        raw_embedding = Embedding(input_dim = self.user_count, output_dim = self.latent_dim, input_length = 1, **kwargs)(input)
        if nonNeg: 
            nonNegConst = NonNeg()(raw_embedding)
            return Flatten()(nonNegConst)
        
        return Flatten()(raw_embedding)
    
    @abc.abstractmethod 
    def _buildMergeLayer(self): raise NotImplementedError