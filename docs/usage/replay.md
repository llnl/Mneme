# Replay

The `mneme replay` command re-executes a previously recorded GPU kernel
in isolation, using the artifacts produced during recording.
For full CLI details, see [CLI → mneme replay](cli.md#mneme-replay).

## Verification

After executing the kernel, Mneme compares the resulting device memory
state against the recorded **epilogue** snapshot.  The replay result
reports `"verified": true` when the comparison passes for every
recorded memory blob.

By default the comparison is **byte-exact**.  If the application
annotated device buffers with `mneme::annotate()` before the recorded
launch, Mneme uses the stored metadata (data type, threshold, norm)
instead.  This allows replay to accept numerically equivalent results
that differ at the bit level — essential when exploring alternative
optimization pipelines or launch configurations.

For a detailed explanation of the annotation API, metadata fields,
threshold semantics, and a complete example, see
**[Usage → Verification](verification.md)**.

## Handling Large Snapshots

When your prologue and epilogue snapshots are large replay startup
time can dominate total replay time. Try setting `--epilogue-format`
to `diff` to minimize data copied from disk and to the device. 
Furthermore, the `--buffer-size` argument allows you to control the
size of the pinned host buffer used to copy data to the device. You
can try values, e.g. `4MiB`, `512KiB`, etc., to tune performance to 
your system; the default is `32MiB`.

Mneme is ultimately bottlenecked by the disk read bandwidth and host to
device memory bandwidth. Ensure your snapshots are stored on a fast
filesystem if possible.
