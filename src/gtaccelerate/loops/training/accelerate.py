from __future__ import annotations

__all__ = ["AccelerateTrainingLoop"]

import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
from accelerate import Accelerator
from coola.utils import str_indent, str_mapping
from gravitorch import constants as ct
from gravitorch.engines import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.loops.training import BaseBasicTrainingLoop
from gravitorch.utils.profilers import BaseProfiler
from torch.nn import Module
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class AccelerateTrainingLoop(BaseBasicTrainingLoop):
    r"""Implements a training loop that uses ``accelerate.Accelerator``
    to train a model.

    Link: https://huggingface.co/docs/accelerate

    Args:
    ----
        accelerator (``accelerate.Accelerator`` or dict or None,
            optional): Specifies the ``accelerate.Accelerator`` object
            or the parameters to instantiate it. Please read the
            ``accelerate.Accelerator`` documentation to know the
            parameters
            https://huggingface.co/docs/accelerate/accelerator.html.
            If ``None``, it will use the default parameters.
            Default: ``None``
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``True``
        tag (str, optional): Specifies the tag which is used to log
            metrics. Default: ``"train"``
        clip_grad (dict or None, optional): Specifies the
            configuration to clip the gradient. If ``None``, no
            gradient clipping is used during the training.
            Default: ``None``
        observer (``BaseLoopObserver`` or dict or None, optional):
            Specifies the loop observer or its configuration.
            If ``None``, the ``NoOpLoopObserver`` is instantiated.
            Default: ``None``
        profiler (``BaseProfiler`` or dict or None, optional):
            Specifies the profiler or its configuration. If ``None``,
            the ``NoOpProfiler`` is instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from gtaccelerate.loops.training import AccelerateTrainingLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = AccelerateTrainingLoop()
        >>> loop
        >>> loop  # doctest:+ELLIPSIS
        AccelerateTrainingLoop(
          (set_grad_to_none): True
          (accelerator): <accelerate.accelerator.Accelerator object at 0x...>
          (tag): train
          (clip_grad_fn): None
          (clip_grad_args): ()
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
        >>> loop.train(engine)
    """

    def __init__(
        self,
        accelerator: Accelerator | dict | None = None,
        set_grad_to_none: bool = True,
        tag: str = "train",
        clip_grad: dict | None = None,
        observer: BaseLoopObserver | dict | None = None,
        profiler: BaseProfiler | dict | None = None,
    ) -> None:
        self._accelerator = self._setup_accelerator(accelerator or {})
        logger.info(f"accelerator state:\n{self._accelerator.state}")
        super().__init__(tag=tag, clip_grad=clip_grad, observer=observer, profiler=profiler)
        self._set_grad_to_none = bool(set_grad_to_none)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "set_grad_to_none": self._set_grad_to_none,
                    "accelerator": self._accelerator,
                    "tag": self._tag,
                    "clip_grad_fn": self._clip_grad_fn,
                    "clip_grad_args": self._clip_grad_args,
                    "observer": self._observer,
                    "profiler": self._profiler,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _prepare_model_optimizer_dataiter(
        self, engine: BaseEngine
    ) -> tuple[Module, Optimizer, Iterable]:
        logger.info("Preparing the model, optimizer, and data iterable...")
        model, optimizer, dataloader = self._accelerator.prepare(
            engine.model,
            engine.optimizer,
            engine.datasource.get_iterable(iter_id=self._tag, engine=engine),
        )
        logger.info("Training model, optimizer, and data iterable have been created")
        return model, optimizer, dataloader

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.trigger_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(self._set_grad_to_none)
        output = model(batch)
        engine.trigger_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        loss = output[ct.LOSS]
        if torch.isnan(loss):
            logger.warning(
                "NaN detected in loss so backpropagation is skipped "
                f"(iteration: {engine.iteration})"
            )
            engine.trigger_event(EngineEvents.TRAIN_ITERATION_COMPLETED)
            return output

        self._accelerator.backward(loss)
        if self._clip_grad_fn:
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.trigger_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        optimizer.step()
        engine.trigger_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output

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

    def _setup_clip_grad(self, clip_grad: dict) -> tuple[Callable | None, tuple]:
        if not clip_grad:
            return None, ()

        name = clip_grad["name"]
        if name == "clip_grad_value":
            clip_grad_fn = self._accelerator.clip_grad_value_
            clip_grad_args = (clip_grad.get("clip_value", 0.25),)
            logger.info(f"clip gradient by value {clip_grad_args[0]}")
            return clip_grad_fn, clip_grad_args
        if name == "clip_grad_norm":
            clip_grad_fn = self._accelerator.clip_grad_norm_
            clip_grad_args = clip_grad.get("max_norm", 1), clip_grad.get("norm_type", 2)
            logger.info(
                f"clip gradient by maximum norm {clip_grad_args[0]} (norm type: "
                f"{clip_grad_args[1]})"
            )
            return clip_grad_fn, clip_grad_args
        raise ValueError(
            f"Incorrect clip grad name ({name}). The valid values are ``clip_grad_value`` "
            "and ``clip_grad_norm``"
        )
