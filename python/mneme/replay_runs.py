"""
mneme.replay_runs

The runs a replay experiment executes, and the chain that orders them.

An experiment replays one recorded kernel under one
:class:`~mneme.mneme_types.ExperimentConfiguration`. It executes a
:class:`RunChain`: an ordered sequence of :class:`ReplayRun` objects, each of
which builds the kernel (builds are shared between runs that ask for the same
:class:`BuildVariant`), launches it, and returns its own typed
:class:`~mneme.mneme_types.RunResult`.

Adding a kind of run means writing one :class:`ReplayRun` subclass and one
registered :class:`~mneme.mneme_types.RunResult` subclass, then appending the
run to the chain.

Public API
----------
VerificationRun, TimingRun:
  The runs of the default chain.
RunChain, default_run_chain:
  The configurable sequence of runs and the chain Mneme uses by default.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, Iterator, Optional, Tuple

from mneme import tracing
from mneme.device import DeviceFunction, DeviceModule
from mneme.llvm.buffer import MemBufferRef
from mneme.llvm.module import ModuleRef
from mneme.mneme_logging import logger
from mneme.mneme_types import (
    BuildStats,
    ExperimentConfiguration,
    RunResult,
    TimingResult,
    VerificationResult,
)
from mneme.profile import GpuKernelTimer, init_profiler

if TYPE_CHECKING:
    from mneme.replay_executor import BaseExecutor


@dataclass(frozen=True)
class BuildVariant:
    """
    How a run wants the kernel built.

    Runs that ask for equal variants share one build within an experiment.

    Attributes
    ----------
    purge_autoinit : bool
        Whether to run the ``purge-autoinit`` pass before ``config.passes``,
        removing the auto-initialization code Clang inserts.
    """

    purge_autoinit: bool


@dataclass
class BuildArtifact:
    """A built kernel: the transformed IR, the device object, and build measurements."""

    variant: BuildVariant
    ir_module: ModuleRef
    object: MemBufferRef
    stats: BuildStats


class LoadedKernel:
    """A kernel loaded on the device, ready to replay against the recorded state."""

    def __init__(
        self,
        name: str,
        function: DeviceFunction,
        config: ExperimentConfiguration,
        prologue,
        epilogue,
    ):
        self.name = name
        self._function = function
        self._config = config
        self._prologue = prologue
        self._epilogue = epilogue

    def launch(self, launches: int) -> None:
        """Restore the recorded prologue and launch the kernel, ``launches`` times."""
        self._function.profile(
            self._config.grid,
            self._config.block,
            self._prologue._state,
            self._epilogue._state,
            self._config.shared_mem,
            launches,
        )

    @property
    def reg_usage(self) -> int:
        return self._function.reg_usage

    @property
    def const_mem(self) -> int:
        return self._function.const_mem

    @property
    def local_mem(self) -> int:
        return self._function.local_mem


class ReplayExperiment:
    """
    One configuration of one recorded kernel, as seen by the runs of a chain.

    It builds the kernel on request, sharing one build per :class:`BuildVariant`,
    and loads builds onto the device.
    """

    def __init__(
        self,
        executor: "BaseExecutor",
        config: ExperimentConfiguration,
        root_ir: ModuleRef,
    ):
        self.executor = executor
        self.config = config
        self.root_ir = root_ir
        self._artifacts: Dict[BuildVariant, BuildArtifact] = {}
        self.most_recent_artifact: Optional[BuildArtifact] = None

    @property
    def kernel_name(self) -> str:
        return self.executor.kernel_descr.kernel_name

    @property
    def noop_verifies(self) -> bool:
        return self.executor.noop_verifies

    def build(self, variant: BuildVariant) -> BuildArtifact:
        """Return the build for ``variant``, building it on first request."""
        artifact = self._artifacts.get(variant)
        if artifact is None:
            artifact = self.executor.build(self.config, self.root_ir, variant)
            self._artifacts[variant] = artifact
        self.most_recent_artifact = artifact
        return artifact

    @contextmanager
    def load(self, artifact: BuildArtifact) -> Iterator[LoadedKernel]:
        """Load ``artifact`` onto the device for the duration of the ``with`` block."""
        with tracing.span("execution.run"):
            with DeviceModule.from_MemBuffer(artifact.object) as module:
                yield LoadedKernel(
                    self.kernel_name,
                    module.get_function(self.kernel_name),
                    self.config,
                    self.executor.prologue,
                    self.executor.epilogue,
                )

    def replay_matches_recording(self) -> bool:
        """Whether the device memory after the last launch matches the recorded epilogue."""
        return self.executor.prologue == self.executor.epilogue


class ReplayRun(ABC):
    """
    One run of a replay experiment.

    Subclasses are frozen dataclasses so that a chain pickles into worker
    processes. ``kind`` names the run and keys its result.
    """

    kind: ClassVar[str]

    def prepare_process(self) -> None:
        """Set up what this run needs once per process, before the device is used."""

    @abstractmethod
    def execute(self, experiment: ReplayExperiment) -> RunResult:
        """Build, launch, and measure the kernel; return this run's result."""

    def blocks_later_runs(self, result: RunResult) -> bool:
        """Whether ``result`` means the runs after this one must not execute."""
        return False


@dataclass(frozen=True)
class VerificationRun(ReplayRun):
    """
    Launch the kernel once and compare the device memory with the recording.

    Attributes
    ----------
    short_circuit : bool
        Whether a failed verification skips the runs after this one.
    """

    kind: ClassVar[str] = VerificationResult.kind

    short_circuit: bool = False

    def execute(self, experiment: ReplayExperiment) -> VerificationResult:
        artifact = experiment.build(BuildVariant(purge_autoinit=False))
        with experiment.load(artifact) as kernel:
            kernel.launch(1)
            with tracing.span("verify.compare") as event:
                verified = experiment.replay_matches_recording()
                event.annotate(
                    verified=verified, noop_verifies=experiment.noop_verifies
                )
        return VerificationResult(
            build=artifact.stats,
            verified=verified,
            noop_verifies=experiment.noop_verifies,
        )

    def blocks_later_runs(self, result: VerificationResult) -> bool:
        return self.short_circuit and not result.verified


@dataclass(frozen=True)
class TimingRun(ReplayRun):
    """
    Time ``warmup + iterations`` launches of the kernel and read its resource usage.

    The kernel is built with ``purge-autoinit``. Every launch is timed, warmup
    launches included.

    Attributes
    ----------
    iterations : int
        Number of measured launches.
    warmup : int
        Number of launches before the measured ones.
    """

    kind: ClassVar[str] = TimingResult.kind

    iterations: int = 3
    warmup: int = 2

    def prepare_process(self) -> None:
        init_profiler()

    def execute(self, experiment: ReplayExperiment) -> TimingResult:
        artifact = experiment.build(BuildVariant(purge_autoinit=True))
        launches = self.iterations + self.warmup
        with experiment.load(artifact) as kernel:
            with GpuKernelTimer(kernel.name, launches) as timer:
                kernel.launch(launches)
            return TimingResult(
                build=artifact.stats,
                exec_time=timer.measurements,
                reg_usage=kernel.reg_usage,
                const_mem_usage=kernel.const_mem,
                local_mem_usage=kernel.local_mem,
            )


@dataclass
class ChainOutcome:
    """
    What executing a run chain produced.

    Attributes
    ----------
    results : dict of str to RunResult
        Results of the runs that executed, by kind, in chain order.
    executed_every_run : bool
        Whether every run of the chain executed, i.e. no run blocked later ones.
    last_artifact : BuildArtifact
        The build used by the last run that executed.
    """

    results: Dict[str, RunResult]
    executed_every_run: bool
    last_artifact: BuildArtifact


@dataclass(frozen=True)
class RunChain:
    """
    The ordered runs every experiment executes.

    Runs execute in order, each inside a trace span named ``execution.<kind>``.
    A run whose result blocks later runs (see :meth:`ReplayRun.blocks_later_runs`)
    stops the chain. Exceptions raised by a run propagate.

    Attributes
    ----------
    runs : tuple of ReplayRun
        The runs, at most one per kind.
    """

    runs: Tuple[ReplayRun, ...]

    def __post_init__(self):
        object.__setattr__(self, "runs", tuple(self.runs))
        if not self.runs:
            raise ValueError("A run chain needs at least one run")
        kinds = [run.kind for run in self.runs]
        duplicates = sorted({kind for kind in kinds if kinds.count(kind) > 1})
        if duplicates:
            raise ValueError(
                f"A run chain holds at most one run per kind; repeated: "
                f"{', '.join(duplicates)}"
            )

    def prepare_process(self) -> None:
        """Prepare the current process for every run of the chain."""
        for run in self.runs:
            run.prepare_process()

    def execute(self, experiment: ReplayExperiment) -> ChainOutcome:
        """Execute the runs in order on ``experiment``."""
        results: Dict[str, RunResult] = {}
        for position, run in enumerate(self.runs):
            with tracing.span(f"execution.{run.kind}", lambda: asdict(run)):
                result = run.execute(experiment)
            results[run.kind] = result
            if run.blocks_later_runs(result):
                skipped = [later.kind for later in self.runs[position + 1 :]]
                if skipped:
                    logger.warning(
                        f"The {run.kind} run of record {experiment.executor.record_id} "
                        f"under config {experiment.config} blocks the later runs. "
                        f"Skipping: {', '.join(skipped)}."
                    )
                    return ChainOutcome(
                        results, False, experiment.most_recent_artifact
                    )
        return ChainOutcome(results, True, experiment.most_recent_artifact)


def default_run_chain(
    iterations: int = 3, warmup: int = 2, short_circuit: bool = False
) -> RunChain:
    """The chain Mneme runs by default: verification, then timing."""
    return RunChain(
        (
            VerificationRun(short_circuit=short_circuit),
            TimingRun(iterations=iterations, warmup=warmup),
        )
    )
