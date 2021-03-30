FAQs
====

.. contents::
   :depth: 1
   :local:

Does RaNNC work with Apex AMP?
------------------------------
Yes.
Convert your model with ``amp.initialize()`` and pass
the resulting model to ``RaNNCModule`` using ``use_amp_master_params=True``.


How to save/load a RaNNC module
-------------------------------

Use ``state_dict()`` of the RaNNC module.
The returned *state_dict* can be saved and loaded, as with PyTorch.
Make sure ``state_dict()`` is called from all ranks.
Otherwise, the call of ``state_dict()`` would be blocked because RaNNC gathers parameters across all ranks.

RaNNCModule also modifies optimizer's ``state_dict()``.
To collect states across all ranks, set ``from_global=True``.
Note that the returned *state_dict* can be loaded after RaNNC partitions the model.
``load_state_dict()`` also needs the keyword argument ``from_global=True``.
You can find typical usages in `examples <https://github.com/nict-wisdom/rannc-examples/>`_.


How to use gradient accumulation
--------------------------------

As default, RaNNC implicitly performs allreduce (sum) of gradients on all ranks after a backward pass.
To prevent this allreduce, you can use ``pyrannc.delay_grad_allreduce(True)``.

After a specified number of forward/backward steps, you can explicitly perform allreduce
using ``allreduce_grads`` of your ``RaNNCModule``.


My model takes too long before partitioning is determined
---------------------------------------------------------

By setting ``save_deployment=true``, RaNNC outputs the deployment state to a file called ``deployment_file`` after
partitioning is determined. You can load the deployment file by setting ``load_deployment=true``.
This greatly saves time if you run a program using RaNNC with similar settings, e.g. with only the learning rate different.
(See also :doc:`config`)

When you are unsure whether the partitioning process is continuing or has already failed, you can change the log level of
the partitioning module. Changing log levels of ``MLPartitioner`` and ``DPStaging`` will show you the progress of the
partitioning process.
(See also :doc:`logging`)


.. Custom cpp functions do not work with RaNNC
.. ---------------------------------------------



.. How should I use a model that takes kwargs?
.. ------------------------------------


.. Does RaNNC work with the torch.distributed package?
.. -----------------------------------------------






