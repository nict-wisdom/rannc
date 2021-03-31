# RaNNC (Rapid Neural Network Connector)

[RaNNC](http://arxiv.org/abs/2103.16063) is automatic parallelization middleware used to train very large-scale neural networks.
Since modern networks often have billions of parameters, they do not fit the memory of GPUs.
RaNNC automatically partitions such a huge network with model parallelism and computes it using multiple GPUs.

- [Documentation](https://nict-wisdom.github.io/rannc/)
- [Examples](https://github.com/nict-wisdom/rannc-examples/)

Compared to existing frameworks, including Megatron-LM and Mesh-TensorFlow,
which require users to implement partitioning of the given network, RaNNC automatically partitions
a network for PyTorch without any modification to its description.
In addition, RaNNC basically has no limitation on its network architecture while the existing frameworks are only applicable to transformer-based networks.

The code below shows a simple usage of RaNNC.
Following the style of PyTorch's data parallelism, RaNNC expects the training script to be launched with an MPI so that
the number of processes matches the number of available GPUs.

```python
model = Net()                  # Define a network
model.to(torch.device("cuda")) # Move paramsters to a cuda device
optimizer = optim.Adam(model.parameters(), lr=0.01) # Define an optimizer
model = pyrannc.RaNNCModule(model, optimizer)  ##### Wrap by RaNNCModule #####
loss = model(input)            # Run a forward pass
loss.backward()                # Run a backward pass
optimizer.step()               # Update parameters
```

You only need to insert the line highlighted above.
RaNNC profiles computation times and memory usage of the components in the network and
determines the partitioning of the network so that each partitioned fragment fits the GPU memory and the training throughput is optimized.
