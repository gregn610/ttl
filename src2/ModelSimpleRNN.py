from keras.layers import Masking, SimpleRNN, Dense
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Nadam

from Consts import CONST_EMPTY
from ModelAbstract import ModelAbstract


class ModelSimpleRNN(ModelAbstract):

    def buildModel(self, batch_size, timesteps, input_dim, in_neurons, hidden_layers, hidden_neurons, out_neurons,
                         rnn_activation, dense_activation):

        self.model = Sequential()

        self.model.add(Masking(mask_value  = CONST_EMPTY,
                          input_shape = ( timesteps, input_dim),
                         )
                 )

        self.model.add(SimpleRNN(in_neurons,
                       activation       = rnn_activation,
                       return_sequences = True,
        #               stateful         = True,
                    ),
                )
        for _ in range(1,hidden_layers):
            self.model.add(SimpleRNN(hidden_neurons,
                           activation        = rnn_activation,
                           return_sequences  = True,
        #                   stateful          = True,
                        ),
                    )

        if hidden_layers > 0 :
            self.model.add(SimpleRNN(out_neurons,
                       activation        = rnn_activation,
                       return_sequences  = False,
        #               stateful         = True,
                    ),
                )
        #self.model.add(Dense(in_neurons,
        #               activation        = dense_activation,
        #               ))
        self.model.add(Dense(1,
                       activation        = dense_activation,
                       ))
#        opt = SGD(lr = 0.05, decay = 0.1, momentum = 0.9, nesterov = True)
        opt = RMSprop(decay=1e-3)

        self.model.compile(loss      = "mean_squared_error",
                      optimizer = opt,
                      metrics   = ['accuracy',]
                    )

        return self.model
