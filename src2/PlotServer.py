import math
from itertools import zip_longest
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.charts import Step
from keras.utils.visualize_util import model_to_dot
import matplotlib.pyplot as plt


class PlotServer(object):

    def visualize_history(history, data_filename=None):
        plt.figure(figsize=(12, 6))
        discard = int(math.ceil(len(history['loss'])/100))*5
        # skip the first few, they destroy plot scale
        plt.plot(history['loss'][discard:], label='loss')
        plt.plot(history['val_loss'][discard:], label='val_loss')

        plt.ylabel('error')
        plt.xlabel('iteration')
        plt.legend()
        #plt.ylim([0, 0.005])
        plt.title('training error')
        if (data_filename is not None):
            plt.savefig(data_filename)
            plt.close()
        else:
            plt.show()


    # In[48]:

    def bokehplot_losses(history):
        bkp = figure(title='Losses',
           plot_width=900,
           x_axis_label='Error',
           y_axis_label='Iteration',
              )

        # skip the first few, they destroy plot scale
        discard = int(math.ceil(len(history['loss'])/100))*5

        bkp.line(
            range(discard, len(history['loss'])),
            history['loss'][discard:],
            legend="Loss",
            line_color='blue'
        )

        bkp.line(
            range(discard, len(history['loss'])),
            history['val_loss'][discard:],
            legend="Val Loss",
            line_color='green'
        )

        bkp.legend.location = "bottom_right"
        # show the results
        show(bkp)


    # In[49]:

    def bokehplot_X_y_yhat( title, batchSample, predictions):
        """
        create a new plot with a title and axis labels
        """
        dfX = batchSample.dfX
        dfy = batchSample.dfy
        dfdy = batchSample.debug_dfdy
        debug_dfX = batchSample.debug_dfX

        bkp = figure(title=title,
               plot_width=900,
               x_axis_label='Log Event',
               y_axis_label='Time',
               y_axis_type="datetime"
                  )
        # add a line renderer with legend and line thickness.
        bkp.circle(
            dfX.index[debug_dfX.__dbg_criticalpath_mask == True].values,
            dfX[batchSample.event_time_col][debug_dfX.__dbg_criticalpath_mask == True].values,
            legend="X Critical Path Log Event",
            size=10,
            line_color='red'
        )
        bkp.asterisk(
            dfX.index[debug_dfX.__dbg_criticalpath_mask == False].values,
            dfX[batchSample.event_time_col][debug_dfX.__dbg_criticalpath_mask == False].values,
            legend="X Noise Log Event",
            size=10,
            line_color='purple'
        )

        bkp.line(
            dfX.index.values,
            dfX[batchSample.event_time_col].values,
            legend="X Log Event",
            line_color='red'
        )

        bkp.line(
            dfX.index.values,
            dfdy['__dbg_realtime_finish'].values,
            legend="Y Realtime Finish",
            line_width=2,
            line_color='black'
        )

        bkp.line(
            dfX.index.values,
            predictions,
            legend="Yhat Prediction",
            line_width=4,
            line_color='green'
        )
        bkp.legend.location = "bottom_right"
        # show the results
        show(bkp)



    #def xxx(self):
    #
    #    SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


    def plotDebugBatchSample(modelData, debugBatchSample):
        Xt, yt = modelData.convert_to_numpy([debugBatchSample, ])
        X, Y = np.mgrid[:Xt.shape[1], :Xt.shape[2]]

        fig = plt.figure(figsize=(12, 6))

        ax11 = fig.add_subplot(2, 2, 1, projection='3d')
        surf11 = ax11.plot_surface(X, Y, Xt[1, :, :])

        ax12 = fig.add_subplot(2, 2, 2, projection='3d')
        surf12 = ax12.plot_surface(X, Y, Xt[22, :, :])

        ax21 = fig.add_subplot(2, 2, 3, projection='3d')
        surf21 = ax21.plot_surface(X, Y, Xt[33, :, :])

        ax22 = fig.add_subplot(2, 2, 4, projection='3d')
        surf22 = ax22.plot_surface(X, Y, Xt[-1, :, :])

        ax11.set_xlabel('timesteps')
        ax11.set_ylabel('features')
        ax11.set_zlabel('Scaled Wshed time')

        plt.show()


    def plotDebugBatchSampleBar(modelData, debugBatchSample):
        Xt, yt = modelData.convert_to_numpy([debugBatchSample, ])
        X, Y = np.mgrid[:Xt.shape[1], :Xt.shape[2]]

        fig = plt.figure(figsize=(12, 6))

        ax = fig.add_subplot(111, projection='3d')
        for c, z in zip_longest(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
            xs = np.arange(20)
            ys = np.random.rand(20)

            # You can provide either a single color or an array. To demonstrate this,
            # the first bar of each set will be colored cyan.
            cs = [c] * len(xs)
            ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


    # In[ ]:

    plotDebugBatchSampleBar(modelData, debugBatchSample)

    # In[ ]:

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np


    def plotPredictions(predictions):
        plt.figure(figsize=(12, 6))
        # skip the first few, they destroy plot scale
        plt.plot(predictions, label='prediction')

        plt.ylabel('time')
        plt.xlabel('iteration')
        plt.legend()
        plt.title('Predictions over nb_samples')
    plt.show()

