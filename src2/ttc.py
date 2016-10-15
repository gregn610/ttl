#!/usr/bin/env python3
"""Time Till Complete.

Usage:
  ttc.py preprocess [--pandas-reader=(csv|excel|json|table)]<modelData.h5> LOGFILES ...
  ttc.py train [--reset] [--gpu-ssh-host=<gpu-ssh-host> [--gpu-ssh-port=<gpu-ssh-port>] [--gpu-ssh-keyfile=<gpu-ssh-keyfile>]]<modelFile.ttc> <modelData.h5>
  ttc.py predict  [--web [--web-port=<web-port>] [--xml]] [--watch [--interval=<seconds>]]<modelFile.ttc> LOGFILE
  ttc.py graph [--web [--web-port=<web-port>] [--xml]]<modelFile.ttc> LOGFILE
  ttc.py (-h | --help)
  ttc.py --version


Options:
  --pandas-reader=(csv|table|excel|json)  Pandas read_ method. [default: csv] See http://pandas.pydata.org/pandas-docs/stable/api.html#input-output
  --gpu-ssh-host=<gpu-ssh-host>           ssh host with the GPU where ttc.py can be run.
  --gpu-ssh-port=<gpu-ssh-port>           ssh port with the GPU where ttc.py can be run [default: 22].
  --gpu-ssh-keyfile=<gpu-ssh-keyfile>     ssh key file [default: ~/.ssh/id_rsa.pub ].
  --watch                                 Monitor the log file for writes and make a prediction
  --interval=<seconds>                    If monitoring how frequent to update prediction [default: 15 ]
  --web                                   Results to an HTTP interface
  --web-port=<web-port>                   The port to use for the HTTP interface [default: 8080]
  --xml                                   No HTML interface, just raw XML
  --reset                                 Overwrite any existing learning
  -h --help     Show this screen.
  --version     Show version.


Commands:
  preprocess  Preprocess historic log files into <modelData.h5>
  train       Read <modelData.h5> and apply the machine learning
  graph    Assess the accuracy of the model.
  predict     Read or monitor a log file and make a prediction or ongoing predictions if in watch mode

"""
import os
from docopt import docopt
import sys

from OutServiceGraphics import OutServiceGraphics

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Time Till Complete 1.0')

    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
    os.environ['CUDA_HOME'] = '/usr/local/cuda'

    if arguments['--xml']:
        from OutServiceXml import OutServiceXml
        out = OutServiceXml()
    else:
        from OutServicePrint import OutServicePrint
        out = OutServicePrint()


    if arguments['preprocess'] == True:
        print('preprocessing log files into %s' % (arguments['<modelData.h5>']), file=sys.stderr)
        from TTCModelData import TTCModelData

        modelData = TTCModelData()

        # Thanks: http://stackoverflow.com/a/4060259
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        fnames = list(map( lambda fn: os.path.join(__location__, fn), arguments['LOGFILES'] ))

        modelData.load_raw_sample_files( fnames )
        modelData.save_np_data_file(arguments['<modelData.h5>'] )


    elif arguments['train'] == True:
        from TTCModelData import TTCModelData
        from ModelSimpleRNN import ModelSimpleRNN

        modelData = TTCModelData()
        modelData.load_np_data_file(arguments['<modelData.h5>'] )

        mlModel = ModelSimpleRNN()
        if os.path.isfile(arguments['<modelFile.ttc>']):
                if arguments['--reset']:
                    print('Overwriting %s' % arguments['<modelFile.ttc>'], file=sys.stderr)
                else:
                    print('Reloading %s' % arguments['<modelFile.ttc>'], file=sys.stderr)
                    mlModel.load_ml_model(arguments['<modelFile.ttc>'])


        mlModel.buildModel(batch_size        = modelData.X_train.shape[1],
                           timesteps         = modelData.X_train.shape[1],
                           input_dim         = modelData.X_train.shape[2],
                           in_neurons        = 333,
                           hidden_layers     = 1,
                           hidden_neurons    = 333,
                           out_neurons       = 333,
                           rnn_activation    = 'tanh',
                           dense_activation  = 'linear'
                           )
        mlModel.train(modelData.X_train, modelData.y_train,
                      modelData.X_validation, modelData.y_validation,
                      epochs = 9,
                      verbose=1
                      )
        print('Saving trained model to: %s' % arguments['<modelFile.ttc>'], file=sys.stderr)
        mlModel.save_ml_model(modelData, arguments['<modelFile.ttc>'])


    elif arguments['graph'] == True:
        # ToDo: add another docopt arg to graph predictions or learning rates or SVG(model_to_dot(...))
        from ModelSimpleRNN import ModelSimpleRNN

        mlModel = ModelSimpleRNN()
        bs, predictions, idx = mlModel.evaluate(arguments['<modelFile.ttc>'], arguments['LOGFILE'])

        out = OutServiceGraphics()
        out.printEvaluation(bs, predictions, data_filename=None)



    elif arguments['predict'] == True:
        # ToDo: Think about watching open file or stdin like tail -f
        from ModelSimpleRNN import ModelSimpleRNN

        mlModel = ModelSimpleRNN()
        predictions = mlModel.predict(arguments['<modelFile.ttc>'], arguments['LOGFILE'])[0]
        out.printPredictions(predictions, ml_model=arguments['<modelFile.ttc>'], sample_file=arguments['LOGFILE'] )

    else:
        print("Arguments:\n%s" % str(arguments), file=sys.stderr)
        #raise Exception # How did we get here?
