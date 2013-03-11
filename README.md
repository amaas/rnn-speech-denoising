rnn-speech-denoising
====================

Recurrent neural network training for noise reduction in robust automatic speech recognition.


Dependencies
====================
The software depends on Mark Schmidt's minFunc package for convex optimization, 
available here: http://www.di.ens.fr/~mschmidt/Software/minFunc.html

Additionally, we have included Mark Hasegawa-Johnson's HTK write and read functions 
that are used to handle the MFCC files.

We used the aurora2 dataset available here: http://aurora.hsnr.de/aurora-2.html


Getting Started
====================
A sample experiment is in train_aurora_local.m. You must change the first 
three paths at the top of the file before you can run it. 
  * codeDir: This directory. Where the drdae code is
  * minFuncDir: Path to the minFunc dependency
  * baseDir: Where you want to run the experiment. As the experiment runs, intermediate 
    models will be saved in a directory. For simplicity, we found it useful to create 
    separate directories for each experiment
There are a number of additional parameters to tune. A few important ones are:
  * dropout: Enable dropout
  * tieWeights: Enable tied weights in the network
  * layerSizes: The sizes of hidden layers in the network and the output layer
  * temporalLayer: Enables temporal connections in the RNN

Once you have all the parameters tuned, run 'matlab -r train_aurora_local.m'

Using Your Own Datasets
====================
The code is written so that you can try out different datasets by just supplying a 
different loader. For an example, see load_aurora.m.