from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers

class actor():

    def __init__(self, state_dim, num_actions, layers=[]):

        self.model = self.create_model(state_dim, num_actions, layers)

        updates = optimizer.get_updates(
            params=self.model.trainable_weights, loss=-K.mean(combined_output))

    def create_model(state_dim, num_actions, layers=[]):
        inputs = Input(shape=(state_dim,))

        if layers != []:
            for units in layers:
                x = Dense(units, activation="sigmoid")(x)
        
            outputs = Dense(actions, activation = "softmax")
        
        else:
            # Binomial regression?
            outputs = Dense(actions, activation= "softmax")

        model = Model(inputs=inputs, outputs=predictions)
        return model.compile(optimizer='rmsprop')

class critic():
