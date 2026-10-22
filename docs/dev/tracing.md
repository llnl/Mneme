# Tracing

Set `MNEME_TRACE_DIR` to an output directory before starting Mneme to enable
tracing. Mneme writes JSON trace files there that you can open in
[Perfetto](https://ui.perfetto.dev/). Tracing is disabled if this environment
variable is not set or empty.
