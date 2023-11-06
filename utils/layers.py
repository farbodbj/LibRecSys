from keras.layers import Dense

def denseSequenceFromList(input_layer, unit_counts: list[int], activation: list|str):
    if type(activation) is list: pass
    else: [activation * len(unit_counts)]
    tmp = input_layer
    for idx, unit_count in enumerate(unit_counts):
        layer = Dense(units = unit_count, activation = activation[idx])
        tmp = layer(tmp)
        
    return tmp
    
        