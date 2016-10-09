def loop_sequence(seq):
    # Thx: http://stackoverflow.com/a/16638648/266387
    while True:
        for idx, elem in enumerate(seq):
            yield (idx, elem)


# In[ ]:

testSampleLooper = loop_sequence(modelData.testing_samples)


# In[ ]:

def run_next_prediction(sampleGenerator, modelData, model):
    (idx, batchSample) = next(sampleGenerator)
    # Convert to a DebugBatchSample class
    debugBatchSample = batchSample
    debugBatchSample.__class__ = DebugBatchSample
    debugBatchSample._conversion_from_BatchSample()

    # This is weak/wrong.
    Xpop, ypop = modelData.convert_to_numpy([debugBatchSample, ])

    print('Xpop.shape : %s' % str(Xpop.shape))

    predictions = []

    # Maybe should predict on batchs Xpop[0:idx,:,:]
    # and mode.reset() for each new Xpop ???

    for sliced in Xpop:
        p = model.predict(sliced[np.newaxis, :, :])
        # print('p: %s' % str(p))
        predictions.append(
            debugBatchSample.regularizedToDateTime(debugBatchSample.event_time_col, (p[0, 0]))
        )

    return (debugBatchSample, predictions, idx)


# In[ ]:

(batchSample, predictions, idx) = run_next_prediction(testSampleLooper, modelData, model)
bokehplot_X_y_yhat("Sample #%d, %s - X vs Y vs Yhat" % (idx,
                                                        batchSample.dfX.loc[0, [batchSample.event_time_col]].values[
                                                            0].strftime("%A")),
                   batchSample,
                   predictions)

# In[ ]:

(batchSample, predictions, idx) = run_next_prediction(testSampleLooper, modelData, model)
bokehplot_X_y_yhat("Sample #%d, %s - X vs Y vs Yhat" % (
    idx,
    batchSample.dfX.loc[0, [batchSample.event_time_col]].values[0].strftime("%A")
),
                   batchSample,
                   predictions,
                   )

(batchSample, predictions, idx) = run_next_prediction(testSampleLooper, modelData, model)
bokehplot_X_y_yhat("Sample #%d, %s - X vs Y vs Yhat" % (
    idx,
    batchSample.dfX.loc[0, [batchSample.event_time_col]].values[0].strftime("%A")
),
                   batchSample,
                   predictions,
                   )

# print(prediction)
## Skip thru some predictions
discard = next(testSampleLooper)
discard = next(testSampleLooper)

# <h2>Unit Tests</h2>

# In[ ]:

(idx, debugBatchSample) = next(testSampleLooper)
# (idx,debugBatchSample) = 0, BatchSample('/home/greg/Projects/Software/TimeTillComplete.com/data/population_v2.1/2006-11-11.log', 0, 1)


# In[ ]:

tmpdfX = debugBatchSample.dfX
print('tmpdfX.shape : %s' % str(tmpdfX.shape))
HTML(tmpdfX.to_html())

# In[ ]:

tmpdfI = debugBatchSample._get_dfI()
print('tmpdfI.shape : %s' % str(tmpdfI.shape))
HTML(tmpdfI.to_html())

# In[ ]:

tmpX, tmpy = modelData.convert_to_numpy([debugBatchSample, ])
print('tmpX.type : %s' % type(tmpX))
print('tmpX.shape: %s' % str(tmpX.shape))

# In[ ]:

sample = 1
df1 = pd.DataFrame(data=tmpX[sample, :, :],
                   index=range(tmpX.shape[1]),
                   columns=range(tmpX.shape[2]))
HTML(df1.to_html())

# In[ ]:

sample = 22
df22 = pd.DataFrame(data=tmpX[sample, :, :],
                    index=range(tmpX.shape[1]),
                    columns=range(tmpX.shape[2]))
HTML(df22.to_html())

# In[ ]:

sample = 33
df33 = pd.DataFrame(data=tmpX[sample, :, :],
                    index=range(tmpX.shape[1]),
                    columns=range(tmpX.shape[2]))
assert tmpX[33:, :, :].all() == CONST_EMPTY
HTML(df33.to_html())

# In[ ]:

sample = -1
dfm1 = pd.DataFrame(data=tmpX[sample, :, :],
                    index=range(tmpX.shape[1]),
                    columns=range(tmpX.shape[2]))
HTML(dfm1.to_html())
