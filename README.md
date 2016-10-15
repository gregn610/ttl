<h1>Time Till Complete</h1>

<h2>Overview</h2>
So this is a utility that uses machine learning to analyze historic log files and
then predict at what time the daily batch is expected to complete.


<h2>Status</h2>
A long long way still to go.


<h2>Built with:</h2>
<ol>
<li>Python 3</li>
<li>Keras</li>
<li>Theano or TensorFlow</li>
<li>Pandas</li>
<li>Numpy</li>
<li>h5py</li>
<li>pytables</li>
<li>tqdm</li>
<li>bokeh</li>
</ol>

<h2>Usage</h2>
*lightly edited*

~~~
./ttc.py
Usage:
  ttc.py preprocess [--pandas-reader=(csv|excel|json|table)] [-q | --quiet] <modelData.h5> LOGFILES ...
  ttc.py train [--reset] [--gpu-ssh-host=<gpu-ssh-host> [--gpu-ssh-port=<gpu-ssh-port>] [--gpu-ssh-keyfile=<gpu-ssh-keyfile>]] [-q | --quiet] <modelFile.ttc> <modelData.h5>
  ttc.py evaluate [--web [--web-port=<web-port>] [--json]] [-q | --quiet] <modelFile.ttc> <modelData.h5>
  ttc.py predict  [--web [--web-port=<web-port>] [--json]] [--watch [--interval=<seconds>]] [-q | --quiet] <modelFile.ttc> LOGFILE
  ttc.py (-h | --help)
  ttc.py --version
~~~

then
~~~
$ ./ttc.py preprocess modeldata.h5 ../data/population_v2.1/*.log
preprocessing log files into modeldata.h5
training files: 100%|███████████████████████████████████████| 1072/1072 [00:23<00:00, 44.98it/s]
validation files: 100%|███████████████████████████████████████| 286/286 [00:06<00:00, 44.74it/s]
testing files: 100%|███████████████████████████████████████| 72/72 [00:01<00:00, 45.29it/s]
~~~


and then ...
~~~
$ ./ttc.py train trained_model.ttc modeldata.h5
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
training_samples: 0it [00:00, ?it/s]
validation_samples: 0it [00:00, ?it/s]
testing_samples: 0it [00:00, ?it/s]
Reloading trained_model.ttc
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties:
name: GeForce GTX 1060 6GB
major: 6 minor: 1 memoryClockRate (GHz) 1.7085
pciBusID 0000:06:00.0
Total memory: 5.93GiB
Free memory: 5.87GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:838] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:06:00.0)
Exception ignored in: 'h5py._errors.set_error_handler'
RuntimeError: Failed to retrieve old handler
Exception ignored in: 'h5py._errors.set_error_handler'
RuntimeError: Failed to retrieve old handler
Train on 58960 samples, validate on 15730 samples
Epoch 1/9
   32/58960 [..............................] - ETA: 1446s - loss: 137978.4062 - acc: 0.0000e+00I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 6195 get requests, put_count=5320 evicted_count=1000 eviction_rate=0.18797 and unsatisfied allocation rate=0.318805
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
  384/58960 [..............................] - ETA: 363s - loss: 126694.3529 - acc: 0.0000e+00I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 3990 get requests, put_count=4251 evicted_count=1000 eviction_rate=0.235239 and unsatisfied allocation rate=0.190977
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 256 to 281
  992/58960 [..............................] - ETA: 302s - loss: 132258.5277 - acc: 0.0000e+00I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 10854 get requests, put_count=10824 evicted_count=1000 eviction_rate=0.0923873 and unsatisfied allocation rate=0.100332
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 655 to 720
58960/58960 [==============================] - 287s - loss: 46194.7196 - acc: 0.0000e+00 - val_loss: 7492.9427 - val_acc: 0.0000e+00
Epoch 2/9
58960/58960 [==============================] - 286s - loss: 4384.3694 - acc: 0.0000e+00 - val_loss: 4078.4203 - val_acc: 0.0000e+00


&lt; snip &gt;


Epoch 8/9
58960/58960 [==============================] - 286s - loss: 3818.6291 - acc: 0.0000e+00 - val_loss: 4017.7344 - val_acc: 0.0000e+00
Epoch 9/9
58960/58960 [==============================] - 286s - loss: 3818.5152 - acc: 0.0000e+00 - val_loss: 4017.4420 - val_acc: 0.0000e+00
Saving trained model to: trained_model.ttc
Exception ignored in: <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x7fb080ac4518>>
Traceback (most recent call last):
  File "/usr/local/lib/python3.4/dist-packages/tensorflow/python/client/session.py", line 522, in __del__
  File "/usr/local/lib/python3.4/dist-packages/tensorflow/python/client/session.py", line 518, in close
AttributeError: 'NoneType' object has no attribute 'raise_exception_on_not_ok_status'
~~~
and then ...
~~~
$ # with stdout redirected to /dev/null
$ ./ttc.py predict modelfile.ttc ../data/population_v2.1/2007-11-11.log 2> /dev/null
[[Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 01:54:49.147797'), Timestamp('2007-11-11 00:00:16.897200')]]
~~~
