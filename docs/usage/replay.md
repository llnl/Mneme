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
optimization pipelines or launch configurations.  `--compare-policy
FILE` overrides the recorded metadata without re-recording.

For a detailed explanation of the annotation API, metadata fields,
threshold semantics, and a complete example, see
**[Usage → Verification](verification.md)**.
