import BaseDeepLearningModel
from keras.layers import concatenate
from overrides import overrides
import utils.layers

class MlpModel(BaseDeepLearningModel):
    def __init__(self, user_count: int, item_count: int, latent_dim: int, unit_counts: list[int], activation: str):
        super().__init__(user_count, item_count, latent_dim)
        self.unit_counts = unit_counts
        self.activation = activation
        
    @overrides 
    def _buildMergeLayer(self):
        self.merge_layer = concatenate([self.user_embedding_layer, self.item_embedding_layer], axis = 1)
        return self.merge_layer

    def _buildMlp(self, input_layer):
        return utils.layers.denseSequenceFromList(input_layer, unit_counts = self.unit_counts, activation = self.activation)
    
    def _buildPartialGraph(self):
        self.user_input_layer = self._buildInputLayer()
        self.item_input_layer = self._buildInputLayer()
        
        self.user_embedding_layer = self._buildEmbedding(nonNeg = False)(self.user_input_layer)
        self.item_embedding_layer = self._buildEmbedding(nonNeg = False)(self.item_input_layer)
        
        self._buildMergeLayer()
        mlpOut = self._buildMlp(self.merge_layer)
        
        return mlpOut