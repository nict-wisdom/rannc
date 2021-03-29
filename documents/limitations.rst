Limitations
===========

Although ``RaNNCModel`` is designed to work in the manner of ``torch.nn.Module``, it has the following limitations.

Control constructs are ignored
------------------------------

RaNNC uses a computation graph produced by PyTorch's `tracing function <https://pytorch.org/docs/stable/generated/torch.jit.trace.html>`_.
As explained in PyTorch's documentation, the tracing function does not record control constructs, including conditional branches and loops.


Arguments and return values
---------------------------

Arguments and outputs of ``RaNNCModel`` must satisfy the following conditions.

- Arguments must be (mini-)batches tensors, whose first dimension corresponds to samples in a mini-batch.
- Keyword arguments are not allowed.
- Outputs must be (mini-)batches tensors or a loss value (scalar tensor).


