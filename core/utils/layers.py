from keras.layers import Dense

def denseSequenceFromList(input_layer, unit_counts: list[int], activation: list|str):
    activation_list = []
    if type(activation) is list: activation_list = activation
    else: activation_list += [activation] * len(unit_counts)
    
    tmp = input_layer
    for idx, unit_count in enumerate(unit_counts):
        layer = Dense(units = unit_count, activation = activation_list[idx])
        tmp = layer(tmp)
        
    return tmp
    
        