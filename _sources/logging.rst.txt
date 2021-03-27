Logging
=======

RaNNC uses `spdlog <https://github.com/gabime/spdlog>`_ and
`spdlog_setup <https://github.com/guangie88/spdlog_setup>`_ for logging.
You can configure logging by a configuration file
placed at ``~/.pyrannc/logging.toml``.

Since RaNNC has loggers associated with internal modules,
you can set a log level for each module.
The below shows an example of the logging configuration file.

.. code-block::

    global_pattern = "[%Y-%m-%d %T.%f] [%L] <%n>: %v"

    # Sinks
    [[sink]]
    name = "console_st"
    type = "stdout_sink_st"

    [[sink]]
    name = "stderr_st"
    type = "color_stdout_sink_st"

    # Loggers
    [[logger]]
    name = "root"
    sinks = ["console_st"]
    level = "info"

    [[logger]]
    name = "RaNNCModule"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "RaNNCProcess"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "GraphLauncher"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "GraphValueStorage"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "GraphUtil"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "Decomposer"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "Decomposition"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "GraphProfiler"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "ParamStorage"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "GraphConnector"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "TorchDriver"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "AllReduceRunner"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "MLPartitioner"
    sinks = ["stderr_st"]
    level = "info"

    [[logger]]
    name = "DPStaging"
    sinks = ["stderr_st"]
    level = "info"
