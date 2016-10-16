from ModelAbstract import ModelAbstract
from Consts import CONST_EMPTY

from keras.layers import Masking, Dense, TimeDistributed, LSTM
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Nadam



class ModelLSTM(ModelAbstract):
    def __init__(self, *args, **kwargs):
        super(ModelLSTM, self).__init__(*args, **kwargs)

    def buildModel(self, batch_size, timesteps, input_dim, in_neurons, hidden_layers, hidden_neurons, out_neurons,
                   rnn_activation, dense_activation):
        self.model = Sequential()
        self.model.add(Masking(
            batch_input_shape=(None, timesteps, input_dim), mask_value=CONST_EMPTY, )
        )
        self.model.add(LSTM(
            in_neurons, input_dim=input_dim, return_sequences=True, activation=rnn_activation)
        )
        self.model.add(TimeDistributed(
            Dense(hidden_neurons))
        )
        self.model.add(LSTM(
            out_neurons, input_dim=hidden_neurons, return_sequences=False, activation=rnn_activation)
        )
        self.model.add(Dense(
            1, activation=dense_activation)
        )

        opt = SGD(lr=0.00025, decay=0.00003333, )
        self.model.compile(loss="mean_squared_error",
                           optimizer=opt,
                           metrics=['accuracy', ]
                           )

        return self.model
