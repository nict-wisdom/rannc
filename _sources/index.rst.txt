.. rannc_doc documentation master file, created by
   sphinx-quickstart on Thu Dec 24 10:23:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RaNNC (Rapid Neural Network Connector)
======================================

RaNNC is an automatic parallelization middleware to train very large-scale neural networks.
Since modern networks often have billions of parameters, they do not fit the memory of GPUs.
RaNNC automatically partitions such a huge network with model parallelism and computes it using multiple GPUs.

Compared to existing frameworks including Megatron-LM and Mesh-TensorFlow,
which require users to implement partitioning of the given network, RaNNC automatically partitions
a network for PyTorch without any modifications to its description.
In addition, RaNNC basically has no limitation on network architectures while such existing frameworks are only applicable to Transformer-based networks.

The code below shows a simple usage of RaNNC.
Following the style of PyTorch's data parallelism, RaNNC expects the training script to be launched with MPI so that
the number of processes matches the number of available GPUs.

.. code-block:: python
   :emphasize-lines: 4

   model = Net()                  # Define a network
   model.to(torch.device("cuda")) # Move paramsters to a cuda device
   optimizer = optim.Adam(model.parameters(), lr=0.01) # Define an optimizer
   model = pyrannc.RaNNCModule(model, optimizer)  ##### Wrap by RaNNCModule #####
   loss = model(input)            # Run a forward pass
   loss.backward()                # Run a backward pass
   optimizer.step()               # Update parameters

The only thing you need is to insert the line highlighted above.
RaNNC profiles computation times and memory usage of components of the network and
determines partitioning of the network so that each partitioned fragment of the network fits to GPU memory
and the training throughput is optimized.

See our IPDPS2021 paper [#f1]_ for the details of RaNNC's partitioning algorithm.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   tutorial
   limitations
   faq
   references
   logging
   build
   config


Reference
=========

.. [#f1] Automatic Graph Partitioning for Very Large-scale Deep Learning, Masahiro Tanaka, Kenjiro Taura, Toshihiro Hanawa and Kentaro Torisawa, In the Proceedings of 35th IEEE International Parallel and Distributed Processing Symposium (IPDPS 2021), May, 2021. (to appear)
