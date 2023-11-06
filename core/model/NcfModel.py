import MlpModel
import GmfModel
from keras.layers import concatenate, Dense

class NcfModel():
    def __init__(self, user_count: int, item_count: int, gmf_dim: int, mlp_dim: int):
        self.mlp_model = MlpModel(user_count, item_count, mlp_dim)
        self.gmf_model = GmfModel(user_count, item_count, gmf_dim)
        self.merge_layer = None
        
    
    def _buildNcfMerge(self):
        self.merge_layer = concatenate([self.mlp_model.merge_layer, self.gmf_model.merge_layer], axis = 1)
        return self.merge_layer
    
    def _buildNeuMfLayer(self):
        return Dense(units = 1, activation = "sigmoid", kernel_initializer = "lecun_uniform")(self.merge_layer)
    
    def _buildModelGraph(self):
        self.mlp_model._buildPartialGraph()
        self.gmf_model._buildParitalGraph()
        
        self._buildNcfMerge()
        return self._buildNeuMfLayer()(self.merge_layer)