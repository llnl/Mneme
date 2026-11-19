"""
Future-like primitives for Mneme asynchronous evaluation.

This module defines :class:`EvalFuture`, a minimal synchronization object used by
the asynchronous replay/tuning infrastructure (e.g., worker pools). It is not a
drop-in replacement for :class:`concurrent.futures.Future`; instead it provides
only what Mneme needs:

  - stable job identity (``job_id``)
  - the submitted :class:`~mneme.mneme_types.ExperimentConfiguration`
  - a blocking :meth:`EvalFuture.result` interface with optional timeout
  - propagation of a successful :class:`~mneme.mneme_types.ExperimentResult` or an error

Thread-safety
-------------
All state transitions are protected by an internal :class:`threading.Condition`.
Producers call :meth:`EvalFuture.set_result` or :meth:`EvalFuture.set_error`;
consumers call :meth:`EvalFuture.done` or :meth:`EvalFuture.result`.
"""

import threading
from typing import Any, Optional, Dict
from mneme.mneme_types import ExperimentConfiguration, ExperimentResult


class EvalFuture:
    """
    Minimal future for a single Mneme evaluation request.

    An EvalFuture represents one submitted evaluation job and acts as the rendezvous
    point between:
      - the submission thread (creating and enqueuing the job), and
      - a worker/monitor thread that eventually completes the job.

    Unlike ``concurrent.futures.Future``, this class intentionally keeps a small API
    surface to avoid additional dependencies and semantics.

    Attributes
    ----------
    job_id : int
        Stable identifier assigned by the submitting executor.
    config : ExperimentConfiguration
        Configuration associated with this evaluation request.
    ir_revision : int
        IR revision captured when this evaluation was submitted.
    ir_data : str | None
        Replacement IR associated with ``ir_revision``.
    """

    def __init__(
        self,
        job_id: int,
        config: ExperimentConfiguration,
        ir_revision: int = 0,
        ir_data: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        job_id : int
            Unique identifier for this evaluation.
        config : ExperimentConfiguration
            Configuration to be evaluated.
        ir_revision : int, optional
            IR revision captured for this evaluation.
        ir_data : str | None, optional
            Replacement IR associated with the captured revision.
        """
        self.job_id = job_id
        self.config = config  # small dict of input params
        self.ir_revision = ir_revision
        self.ir_data = ir_data

        self._cond = threading.Condition()
        self._completion_events: set[threading.Event] = set()
        self._done = False
        self._result: Optional[ExperimentResult] = None
        self._error: Optional[str] = None

    def set_result(self, result: ExperimentResult):
        """
        Mark the future as completed successfully.

        Parameters
        ----------
        result : ExperimentResult
            Result to publish to waiters.

        Notes
        -----
        This method wakes all threads waiting in :meth:`result`.
        """
        with self._cond:
            self._result = result
            self._complete()

    def set_error(self, error: str):
        """
        Mark the future as completed with an error.

        Parameters
        ----------
        error : str
            Human-readable error message.

        Notes
        -----
        This method wakes all threads waiting in :meth:`result`.
        """
        with self._cond:
            self._error = error
            self._complete()

    def _complete(self):
        self._done = True
        self._cond.notify_all()
        for event in self._completion_events:
            event.set()
        self._completion_events.clear()

    def notify_when_done(self, event: threading.Event):
        """Set the event when this future completes, or immediately if it is done."""
        with self._cond:
            if self._done:
                event.set()
            else:
                self._completion_events.add(event)

    def done(self):
        """
        Return whether the future has completed.

        Returns
        -------
        bool
            ``True`` if either a result or an error has been set.
        """
        with self._cond:
            return self._done

    def result(self, timeout=None) -> Optional[ExperimentResult]:
        """
        Wait for completion and return the evaluation result.

        Parameters
        ----------
        timeout : float | None, optional
            Maximum time in seconds to wait. If ``None``, wait indefinitely.

        Returns
        -------
        ExperimentResult | None
            The published result if available. If a timeout expires before completion,
            the method returns ``None``.

        Raises
        ------
        RuntimeError
            If the evaluation completed with an error (set via :meth:`set_error`).

        Notes
        -----
        Mneme uses :class:`ExperimentResult` as the canonical result container.
        When an error occurs and a partial result exists, the error string is attached
        to the stored result before raising.
        """
        with self._cond:
            if not self._done:
                self._cond.wait(timeout=timeout)

            if self._error:
                if self._result is None:
                    return ExperimentResult(
                        error=self._error, failed=True, executed=False
                    )
                else:
                    self._result.error = self._error

                raise RuntimeError(self._error)

            return self._result
