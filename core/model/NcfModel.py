import tensorflow as tf
from keras.layers import concatenate, Dense, Input
from core.model.MlpModel import MlpModel
from core.model.GmfModel import GmfModel

class NcfModel():
    def __init__(self, user_count: int, item_count: int, gmf_dim: int, mlp_dim: int):
        self.mlp_model = MlpModel(user_count, item_count, mlp_dim, [64, 32, 16], "relu")
        self.gmf_model = GmfModel(user_count, item_count, gmf_dim)
        self.merge_layer = None
        self.user_input = None 
        self.item_input = None
        
    def _buildInputLayer(self):
        return Input(shape = [1], dtype = tf.int32)
    
    def _buildNcfMerge(self):
        self.merge_layer = concatenate([self.mlp_model.output_layer, self.gmf_model.merge_layer], axis = 1)
        return self.merge_layer
    
    def _buildNeuMfLayer(self):
        return Dense(units = 1, activation = "sigmoid", kernel_initializer = "lecun_uniform")(self.merge_layer)
    
    def _buildModelGraph(self):
        self.user_input = self._buildInputLayer()
        self.item_input = self._buildInputLayer()        
        
        self.gmf_model.user_input = self.user_input
        self.gmf_model.item_input = self.item_input
        
        self.mlp_model.user_input = self.user_input
        self.mlp_model.item_input = self.item_input
        
        self.mlp_model._buildPartialGraph()
        self.gmf_model._buildPartialGraph()
        
        self._buildNcfMerge()
        return self._buildNeuMfLayer()