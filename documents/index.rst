.. rannc_doc documentation master file, created by
   sphinx-quickstart on Thu Dec 24 10:23:00 2020.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

RaNNC (Rapid Neural Network Connector)
======================================

RaNNC is an automatic parallelization middleware for training very large-scale neural networks.
Since modern networks often have billions of parameters, they do not fit the memory of GPUs.
RaNNC automatically partitions such a huge network with model parallelism and computes it using multiple GPUs.

Compared to existing frameworks, including `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ and
`Mesh-TensorFlow <https://github.com/tensorflow/mesh>`_,
which require users to implement partitioning of the given network, RaNNC automatically partitions
a network for PyTorch without any modification to its description.
In addition, RaNNC basically has no limitation on its network architecture while the existing frameworks are only applicable to transformer-based networks.

The code below shows a simple usage of RaNNC.
You only need to insert the line highlighted below.

.. code-block:: python
   :emphasize-lines: 4

   model = Net()                  # Define a network
   model.to(torch.device("cuda")) # Move paramsters to a cuda device
   optimizer = optim.Adam(model.parameters(), lr=0.01) # Define an optimizer
   model = pyrannc.RaNNCModule(model, optimizer)  ##### Wrap by RaNNCModule #####
   loss = model(input)            # Run a forward pass
   loss.backward()                # Run a backward pass
   optimizer.step()               # Update parameters


Models used with RaNNC do not need special operators for distributed computation or annotations for partitioning
(See our examples:
`model for the tutorial <https://github.com/nict-wisdom/rannc/blob/main/examples/tutorial.py>`_,
enlarged versions of `BERT <https://github.com/nict-wisdom/rannc-examples/blob/main/bert/modeling.py>`_ and
`ResNet <https://github.com/nict-wisdom/rannc-examples/blob/main/resnet/resnet_wf.py>`_).
RaNNC automatically partitions a model to `subcomponents` so that each subcomponent fits to the GPU memory and a high
training throughput is achieved.

In contrast, for example, Megatron-LM needs special operators like ``ColumnParallelLinear`` and  ``RowParallelLinear``
(See an `example <https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/megatron/model/transformer.py#L59>`_ in Transformer).
Implementing a model using such operators is very hard even for experts because the user needs to consider computational loads
of the model, memory usages, and communication overheads.
In addition, some existing frameworks including Megatron-LM can be applicable only to Transformer family networks.

We confirmed that RaNNC can train a BERT model with approximately 100 billion parameters without a manual
modification/optimization of the definition of the network for model partitioning.

See our IPDPS2021 paper [#f1]_ for the details of RaNNC's partitioning algorithm (`preprint <http://arxiv.org/abs/2103.16063>`_).

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   tutorial
   limitations
   faq
   references
   logging
   config
   build


Reference
=========

.. [#f1] Automatic Graph Partitioning for Very Large-scale Deep Learning, Masahiro Tanaka, Kenjiro Taura, Toshihiro Hanawa and Kentaro Torisawa, In the Proceedings of 35th IEEE International Parallel and Distributed Processing Symposium (IPDPS 2021), pp. 1004-1013, May, 2021.
