# Tuning recorded kernels

Tuning searches for a faster replay of one recorded instance while preserving its recorded behavior. It needs a real recording and a baseline that verifies. Running the original application successfully is not enough.

The [tuning concepts](https://software.llnl.gov/Mneme/concepts/tune/) distinguish the replay configuration, the search space, and the optimizer. Check the installed `mneme tune --help` before choosing a built-in route. Use a Python driver when the available command cannot express the intended search.

## SearchSpace

SearchSpace describes tunable parameters through `dimensions()`, filters samples through `constraints(params)`, maps them to an ExperimentConfiguration through `derived(params)`, and supplies a reference through `baseline()`. Verify with installed version.

Search parameters need not match replay fields directly. A sampled block size might determine a grid size; a fraction might select a discrete launch bound. Generate legal dependent settings where practical. Multiples of 64 and a ceiling of 1024 in an example are not universal rules: use the device limits, all three block dimensions, and the kernel's requirements.

Start with a small parameter family justified by the workload. Keep unrelated settings fixed. Many kernels cannot change block shape safely, so inspect indexing and synchronization before trying a block-size search.

## Scoring

The documented executor call is `AsyncReplayExecutor.evaluate(config)`. Check configuration validity, execution status, `result.verified`, and timing samples before scoring. Baseline and candidate samples must be nonempty, finite, and positive; retain their units.

Minimize execution time or maximize baseline time divided by candidate time. Match the study direction to the objective and mark failed trials as failures so they cannot become winners. Stop if no valid trial completes. Compilation errors and invalid candidates may be isolated by workers, but repeated baseline or device failures need investigation rather than endless retries.

Start with one worker on an allocated GPU. Bound the trial count and runtime, reserve time to save results, and clean up the executor on exceptions. Concurrent work on the same GPU can distort measurements. Remeasure the winner against the baseline under comparable conditions before claiming a speedup.

## Using the supplied script

[scripts/tune_threads.py](../scripts/tune_threads.py) shows the recorded execution loading, SearchSpace mapping, Optuna sampling, and executor cleanup. It does not explicitly validate timing samples before calculating their mean.

The script assumes one-dimensional blocks, preserves the recorded grid, and searches several parameter families at once. Use it as an API example, or narrow it for an initial experiment. Its upper bound of 1024 still needs checking against the target GPU and kernel.

```sh
python scripts/tune_threads.py --record-db <database.json> --record-id <instance-id> --num-trials 10
```


## Saving results

Save the recording database path and selected instance ID so the tuned kernel can be identified. Then also record the relevant objectives and parameters used for the experiment such as the GPU model, search space, and objective used for the experiment. Also keep the baseline and best verified configurations with their timing samples and the number of failed trials.
