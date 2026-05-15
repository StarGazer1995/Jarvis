"""
Core Fault Injection Framework

Provides the foundational classes for injecting faults into mock objects
during testing. Supports both sync and async callables, configurable
failure counts, and composable fault patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


@dataclass
class FaultConfig:
    """Configuration for a single fault injection scenario.

    Attributes:
        exception: The exception class to raise.
        args: Positional arguments passed to the exception constructor.
        kwargs: Keyword arguments passed to the exception constructor.
        fail_count: How many consecutive calls should fail before succeeding.
        success_value: The value returned after failures are exhausted.
            If None and no more faults are configured, the mock will
            return a default value.
    """

    exception: Type[Exception]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    fail_count: int = 1
    success_value: Any = None


class FaultPattern:
    """A composable sequence of fault configurations.

    Allows chaining multiple fault scenarios together, e.g.:
        - Fail with timeout 2 times
        - Then fail with rate limit 1 time
        - Then succeed
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._configs: List[FaultConfig] = []
        self._final_success: Any = None

    def add_fault(
        self,
        exception: Type[Exception],
        fail_count: int = 1,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "FaultPattern":
        """Add a fault to the pattern.

        Args:
            exception: Exception class to raise.
            fail_count: How many consecutive times to raise this exception.
            args: Positional arguments for the exception constructor.
            kwargs: Keyword arguments for the exception constructor.
        """
        self._configs.append(
            FaultConfig(
                exception=exception,
                args=args or (),
                kwargs=kwargs or {},
                fail_count=fail_count,
            )
        )
        return self

    def then_succeed(self, value: Any = None) -> "FaultPattern":
        """Set the final success value after all faults are exhausted.

        Args:
            value: The value to return on success.
        """
        self._final_success = value
        return self

    def build(self) -> Callable:
        """Build the fault injector from this pattern."""
        injector = FaultInjector(*self._configs)
        injector.set_success(self._final_success)
        return injector.build()


class FaultInjector:
    """Creates callable side_effect functions for unittest.mock.

    Injects faults by raising exceptions a configured number of times,
    then optionally returning a success value. Handles both sync and
    async callables automatically.

    Examples:
        # Single fault with positional args, then succeed
        mock.method.side_effect = FaultInjector(
            FaultConfig(ValueError, args=("bad",), fail_count=2),
            success_value="ok"
        ).build()

        # Single fault with keyword args (for project exceptions)
        mock.method.side_effect = FaultInjector(
            FaultConfig(LLMAPIError, kwargs={"message": "boom"}, fail_count=2),
            success_value="ok"
        ).build()

        # Chain multiple fault types
        pattern = FaultPattern()
        pattern.add_fault(TimeoutError, fail_count=2, args=("timed out",))
        pattern.add_fault(ConnectionError, fail_count=1, args=("refused",))
        pattern.then_succeed("done")
        mock.method.side_effect = pattern.build()
    """

    def __init__(
        self,
        *configs: FaultConfig,
        success_value: Any = None,
    ):
        """Initialize the fault injector.

        Args:
            *configs: FaultConfig objects defining the fault sequence.
            success_value: Default value returned after all faults exhausted.
        """
        self._phases: List[_FaultPhase] = []
        self._success_value = success_value

        for config in configs:
            if isinstance(config, FaultConfig):
                self._add_phase(
                    config.exception,
                    config.args,
                    config.kwargs,
                    config.fail_count,
                )
                # Use success_value from last FaultConfig if not explicitly set
                if config.success_value is not None and self._success_value is None:
                    self._success_value = config.success_value
            else:
                raise TypeError(f"Expected FaultConfig, got {type(config)}")

    def _add_phase(
        self,
        exception_cls: Type[Exception],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        count: int,
    ) -> None:
        """Add a fault phase to the sequence."""
        if count < 1:
            raise ValueError("fail_count must be >= 1")
        self._phases.append(_FaultPhase(exception_cls, args, kwargs, count))

    def set_success(self, value: Any) -> "FaultInjector":
        """Set the final success value.

        Args:
            value: Value to return after all faults are exhausted.
        """
        self._success_value = value
        return self

    def build(self) -> Callable:
        """Build the side_effect callable.

        Returns:
            A callable suitable for use as ``mock.method.side_effect``.
            Handles both sync and async contexts automatically.
        """
        phases = list(self._phases)
        success_value = self._success_value

        if not phases and success_value is not None:
            # No faults, just return success (useful for dynamic patterns)
            return lambda *a, **kw: success_value

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            """The actual side_effect function.

            Each call increments call_count for the current phase.
            If call_count exceeds the phase's fail_count, the phase is
            consumed and we recurse to let the next phase handle the call.
            This ensures a phase with fail_count=N fails exactly N times
            before the next phase takes over.
            """
            if not phases:
                if success_value is not None:
                    return success_value
                raise RuntimeError(
                    "Fault injector exhausted — no more fault phases configured. "
                    "Set success_value or add more phases."
                )

            phase = phases[0]
            phase.call_count += 1

            if phase.call_count > phase.fail_count:
                # This phase has delivered all its faults.
                # Remove it, then decide what to do with this call.
                phases.pop(0)
                if not phases and success_value is None:
                    # Perpetual failure mode: keep raising the last phase's exception
                    exc = _construct_exception(phase)
                    raise exc
                if not phases:
                    return success_value
                # Move to next phase — this call is the first of the next phase
                return _side_effect(*args, **kwargs)

            # Raise the configured exception for this fault
            exc = _construct_exception(phase)
            raise exc

        return _side_effect


class _FaultPhase:
    """Internal representation of a single fault phase."""

    __slots__ = ("exception_cls", "args", "kwargs", "fail_count", "call_count")

    def __init__(
        self,
        exception_cls: Type[Exception],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        fail_count: int,
    ):
        self.exception_cls = exception_cls
        self.args = args
        self.kwargs = dict(kwargs)
        self.fail_count = fail_count
        self.call_count = 0


def _construct_exception(phase: _FaultPhase) -> Exception:
    """Construct an exception from a fault phase, trying kwargs then args."""
    try:
        return phase.exception_cls(*phase.args, **phase.kwargs)
    except TypeError:
        # Fallback: some exceptions require positional args only
        return phase.exception_cls(*phase.args)


# Convenience aliases
def build_side_effect(
    exception: Type[Exception],
    fail_count: int = 1,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    success_value: Any = None,
) -> Callable:
    """Quick-build a fault injector side_effect.

    One-liner for simple scenarios:

        mock.method.side_effect = build_side_effect(
            LLMAPIError,
            fail_count=2,
            kwargs={"message": "boom"},
            success_value="ok",
        )
    """
    return FaultInjector(
        FaultConfig(
            exception=exception,
            args=args or (),
            kwargs=kwargs or {},
            fail_count=fail_count,
            success_value=success_value,
        )
    ).build()
