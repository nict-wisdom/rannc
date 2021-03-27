FAQs
====

.. contents::
   :depth: 1
   :local:

Does RaNNC work with Apex AMP?
------------------------------
Yes.
Convert your model with ``amp.initialize()`` and pass
the resulting model to ``RaNNCModule`` with ``use_amp_master_params=True``.


How to save/load a RaNNC module
-------------------------------

Use ``state_dict()`` of the RaNNC module.
The returned *state_dict* can be saved and loaded as with PyTorch.

Please make sure ``state_dict()`` must be called from all ranks.
Otherwise, the call of ``state_dict()`` is blocked because RaNNC gathers parameters across all ranks.


How to use gradient accumulation
--------------------------------

As default, RaNNC implicitly performs allreduce (sum) of gradients on all ranks after a backward pass.
To prevent the allreduce, you can use ``pyrannc.delay_grad_allreduce(False)``.

After a specified number of forward/backward steps, you can explicitly perform allreduce
with ``allreduce_grads`` of your ``RaNNCModule``.


My model takes too long until partitioning is determined
--------------------------------------------------------

By setting ``save_deployment=true``, RaNNC outputs the deployment state to a file ``deployment_file`` after
partitioning is determined. You can load the deployment file by setting ``load_deployment=true``.
This greatly save your time if you run a program using RaNNC with similar settings, e.g. with different learning rate.
(See also :doc:`config`)

When you are unsure that partitioning process keeps going or already failed, you can change the log level of
the partitioning module. Changing log levels of ``MLPartitioner`` and ``DPStaging`` will show you the progress of
partitioning process.
(See also :doc:`logging`)


.. Custom cpp functions does not work with RaNNC
.. ---------------------------------------------



.. How to use a model that takes kwargs
.. ------------------------------------


.. Does RaNNC work with torch.distributed package?
.. -----------------------------------------------


.. How to save/restore optimizer's state
.. -------------------------------------

