from __future__ import annotations

__all__ = ["AccelerateEvaluationLoop"]

import logging
from collections.abc import Iterable
from typing import Any

import torch
from accelerate import Accelerator
from coola.utils import str_indent, str_mapping
from gravitorch.engines import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.evaluation import BaseBasicEvaluationLoop
from gravitorch.loops.evaluation.conditions import BaseEvalCondition
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.utils.profilers import BaseProfiler
from torch.nn import Module

logger = logging.getLogger(__name__)


class AccelerateEvaluationLoop(BaseBasicEvaluationLoop):
    r"""Implements an evaluation loop that uses
    ``accelerate.Accelerator`` to evaluate a model.

    Link: https://huggingface.co/docs/accelerate

    Args:
    ----
        accelerator (``accelerate.Accelerate`` or dict or None,
            optional): Specifies the ``accelerate.Accelerate`` object
            or the parameters to instantiate it. Please read the
            ``accelerate.Accelerator`` documentation to know the
            parameters https://huggingface.co/docs/accelerate/accelerator.html.
            If ``None``, it will use the default parameters.
            Default: ``None``
        tag (str, optional): Specifies the tag which is used to log
            metrics. Default: ``"eval"``
        grad_enabled (bool, optional): Specifies if the gradient is
            computed or not in the evaluation loop. By default, the
            gradient is not computed to reduce the memory footprint.
            Default: ``False``
        condition (``BaseEvalCondition`` or dict or None): Specifies
            the condition to evaluate the loop or its configuration.
            If ``None``, the ``EveryEpochEvalCondition(every=1)`` is
            used.  Default ``None``
        observer (``BaseLoopObserver`` or dict or None, optional):
            Specifies the loop observer or its configuration.
            If ``None``, the ``NoOpLoopObserver`` is instantiated.
            Default: ``None``
        profiler (``BaseProfiler`` or dict or None, optional):
            Specifies the profiler or its configuration. If ``None``,
            the ``NoOpProfiler`` is instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from gtaccelerate.loops.evaluation import AccelerateEvaluationLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = AccelerateEvaluationLoop()
        >>> loop  # doctest:+ELLIPSIS
        AccelerateEvaluationLoop(
          (accelerator): <accelerate.accelerator.Accelerator object at 0x...>
          (grad_enabled): False
          (tag): eval
          (condition): EveryEpochEvalCondition(every=1)
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
        >>> loop.eval(engine)
    """

    def __init__(
        self,
        accelerator: Accelerator | dict | None = None,
        tag: str = "eval",
        grad_enabled: bool = False,
        condition: BaseEvalCondition | dict | None = None,
        observer: BaseLoopObserver | dict | None = None,
        profiler: BaseProfiler | dict | None = None,
    ) -> None:
        self._accelerator = self._setup_accelerator(accelerator or {})
        logger.info(f"accelerator state:\n{self._accelerator.state}")
        super().__init__(tag=tag, condition=condition, observer=observer, profiler=profiler)
        self._grad_enabled = grad_enabled

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "accelerator": self._accelerator,
                    "grad_enabled": self._grad_enabled,
                    "tag": self._tag,
                    "condition": self._condition,
                    "observer": self._observer,
                    "profiler": self._profiler,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _eval_one_batch(self, engine: BaseEngine, model: Module, batch: Any) -> dict:
        engine.trigger_event(EngineEvents.EVAL_ITERATION_STARTED)
        with torch.set_grad_enabled(self._grad_enabled):
            output = model(batch)
        engine.trigger_event(EngineEvents.EVAL_ITERATION_COMPLETED)
        return output

    def _prepare_model_dataiter(self, engine: BaseEngine) -> tuple[Module, Iterable]:
        logger.info("Preparing the model and data iterable...")
        model, dataloader = self._accelerator.prepare(
            engine.model,
            engine.datasource.get_iterable(iter_id=self._tag, engine=engine),
        )
        logger.info("Evaluation model and data iterable have been prepared")
        return model, dataloader

    def _setup_accelerator(self, accelerator: Accelerator | dict) -> Accelerator:
        r"""Sets up the accelerator.

        Args:
        ----
            accelerator (``accelerate.Accelerator`` or dict, optional):
                Specifies the ``accelerate.Accelerator`` object or the
                parameters to instantiate it. Please read the
                ``accelerate.Accelerator`` documentation to know the
                parameters https://huggingface.co/docs/accelerate/accelerator.html.

        Returns:
        -------
            ``accelerate.Accelerator``: The accelerator object.

        Raises:
        ------
            RuntimeError: if the accelerate package is not installed.
        """
        if isinstance(accelerator, Accelerator):
            return accelerator
        logger.info(f"accelerator options: {accelerator}")
        return Accelerator(**accelerator)
