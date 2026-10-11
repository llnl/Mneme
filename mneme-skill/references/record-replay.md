# Recording and replay

Mneme captures GPU kernel executions so they can be replayed separately from the application. Build-time instrumentation preserves the kernel IR and the recording records the launch and device memory state. The replay restores the input state, recompiles and executes the kernel, then checks the resulting memory against the recording.

This follows the supplied [concepts overview](https://software.llnl.gov/Mneme/concepts/overview/) and [replay configuration](https://software.llnl.gov/Mneme/concepts/replay-configuration/) documentation. Confirm details against the user's checkout.

## Prep

Identify the actual build target, input, and correctness check. For supported CMake projects, use `find_package(mneme REQUIRED)` and `add_mneme(target)` as demonstrated by the matching repository example. Build with the compatible compiler and target architecture, then run a bounded input in a GPU allocation. Confirm the application's own correctness check before recording.

### Cuda Vec_add example
For the CUDA vec_add example, the executable accepts a positive integer in units of 10,000 elements. An argument of `10` processes 100,000 elements and should print `Correct output`.

## Recording

Inspect `mneme record --help`, create a fresh output directory, and record the instrumented application with its normal arguments. The supplied CLI uses:

```sh
mneme record -rdb <record-directory> -- <application> <arguments>
```

Replace the placeholders before running. Recording takes an output directory, whereas replay takes a particular database file. For distributed applications, check rank selection and how Mneme must be combined with the site's launcher.

Inspect the generated databases to find the captured kernels and instances. Keep each database together with the IR and memory snapshots it references. Static hashes identify kernels and dynamic hashes distinguish recorded instances; select an actual instance from the database rather than guessing an ID. Successful application execution alone does not establish that any kernels were recorded.

## Replaying

Check `mneme replay --help`. With the supplied CLI, a baseline command has this form:

```sh
mneme replay -rdb <database.json> -rid <instance-id> 'default<O3>'
```

The pass pipeline is positional and needs shell quoting. Omitted launch dimensions and shared memory use the recorded values; other options have their own defaults. Require baseline verification before introducing changes. If it fails, investigate the recording, environment, and replay configuration before tuning.

Replay configurations can control grid and block dimensions, dynamic shared memory, specialization, LLVM passes, backend optimization, and launch bounds. `passes` and `codegen_opt` affect different compilation stages. Only use flags exposed by installed help; a Python configuration field does not necessarily have a CLI option.

Geometry changes must respect kernel indexing, synchronization, and device limits. Count all block dimensions when calculating threads per block. When launch bounds are enabled, `max_threads` must accommodate that total. Specialization uses information from the recorded invocation; do not assume it generalizes to other inputs.

The documented Python configuration provides `is_valid()`, `ground()`, `to_dict()`, and `hash()`. Check their installed semantics. Normalization can remove inactive launch-bound settings, but does not prove a configuration is correct for the kernel. The supplied documentation requires `prune` and `internalize` to be true; do not expose them as tuning choices without confirming support in the installed revision.
