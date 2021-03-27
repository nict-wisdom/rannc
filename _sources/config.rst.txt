Configurations
==============

RaNNC's runtime configurations can be set in the following two ways:

- *Config file*: RaNNC automatically loads a configuration file at ``~/.pyrannc/rannc_conf.toml``. Names of configuration items must be in lower case. The path to the configuration file can be set by an environment variable ``RANNC_CONF_DIR``.
- *Environment variables*: You can overwrite configuration by setting environment variables. Names of variables follows ``RANNC_<CONF_ITEM_NAME>`` in upper case. For example, you can set `mem_margin` in the following table by a variable ``RANNC_MEM_MARGIN``.


.. list-table:: Configurations
   :widths: 20 10 70
   :header-rows: 1

   * - Name
     - Default
     -
   * - mem_margin
     - 0.1
     - Memory margin for model partitioning.
   * - save_deployment
     - true
     - Save deployment of a partitioned model if set to true.
   * - load_deployment
     - false
     - Load deployment of a partitioned model if set to true.
   * - deployment_file
     - ``/tmp/rannc_deployment.bin``
     - Path of deployment file to save/load.
   * - min_pipeline
     - 1
     - Minimum number of microbatches for pipeline parallelism
   * - max_pipeline
     - 32
     - Maxmum number of microbatches for pipeline parallelism
   * - opt_param_factor
     - 2
     - Factor to estimate memory usage by an optimizer. For example, Set this item to 2 for Adam because the optimizer uses two internal data `v` and `s`, whose sizes are equivalent to parameter tensors.
   * - trace_events
     - false
     - Trace internal events if set to true. When true, the event tracing significantly degrades performance.
   * - event_trace_file
     - ``/tmp/rannc_event_trace.json``
     - Path to an event trace file.

The following is an example of the configuration file.

.. code-block::

   opt_param_factor=2
   mem_margin=0.1
   min_pipeline=1
   max_pipeline=4
   save_deployment=true
   load_deployment=false
   trace_events=false
   verify_recomp=false

