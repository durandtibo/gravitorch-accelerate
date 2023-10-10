from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.loops.observers import (
    BaseLoopObserver,
    NoOpLoopObserver,
    PyTorchBatchSaver,
)
from gravitorch.testing import (
    DummyClassificationModel,
    DummyDataSource,
    DummyIterableDataset,
    create_dummy_engine,
)
from gravitorch.utils.events import GEventHandler
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.history import EmptyHistoryError, MinScalarHistory
from gravitorch.utils.profilers import BaseProfiler, NoOpProfiler, PyTorchProfiler
from pytest import fixture, mark, raises
from torch.nn import Module
from torch.optim import SGD, Optimizer

from gtaccelerate.loops.training import AccelerateTrainingLoop


@fixture(autouse=True)
def reset_accelerate_state() -> None:
    AcceleratorState._reset_state(reset_partial_state=True)


def increment_epoch_handler(engine: BaseEngine) -> None:
    engine.increment_epoch(2)


def test_accelerate_training_loop_str() -> None:
    assert str(AccelerateTrainingLoop()).startswith("AccelerateTrainingLoop(")


def test_accelerate_training_loop_accelerator_none() -> None:
    assert isinstance(AccelerateTrainingLoop()._accelerator, Accelerator)


def test_accelerate_training_loop_accelerator_object() -> None:
    training_loop = AccelerateTrainingLoop(accelerator=Accelerator(cpu=True))
    assert isinstance(training_loop._accelerator, Accelerator)
    assert training_loop._accelerator.state.device.type == "cpu"


def test_accelerate_training_loop_accelerator_cpu_only() -> None:
    assert AccelerateTrainingLoop(accelerator={"cpu": True})._accelerator.state.device.type == "cpu"


@mark.parametrize("set_grad_to_none", (True, False))
def test_accelerate_training_loop_set_grad_to_none(set_grad_to_none: bool) -> None:
    assert (
        AccelerateTrainingLoop(set_grad_to_none=set_grad_to_none)._set_grad_to_none
        == set_grad_to_none
    )


def test_accelerate_training_loop_set_grad_to_none_default() -> None:
    assert AccelerateTrainingLoop()._set_grad_to_none


@mark.parametrize("tag", ("pre-training", "custom name"))
def test_accelerate_training_loop_tag(tag: str) -> None:
    assert AccelerateTrainingLoop(tag=tag)._tag == tag


def test_accelerate_training_loop_tag_default() -> None:
    assert AccelerateTrainingLoop()._tag == "train"


def test_accelerate_training_loop_clip_grad_none() -> None:
    training_loop = AccelerateTrainingLoop()
    assert training_loop._clip_grad_fn is None
    assert training_loop._clip_grad_args == ()


def test_accelerate_training_loop_clip_grad_clip_grad_value_without_clip_value() -> None:
    training_loop = AccelerateTrainingLoop(clip_grad={"name": "clip_grad_value"})
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (0.25,)


@mark.parametrize("clip_value", (0.1, 1))
def test_accelerate_training_loop_clip_grad_clip_grad_value_with_clip_value(
    clip_value: float,
) -> None:
    training_loop = AccelerateTrainingLoop(
        clip_grad={"name": "clip_grad_value", "clip_value": clip_value}
    )
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (clip_value,)


def test_accelerate_training_loop_clip_grad_clip_grad_norm_without_max_norm_and_norm_type() -> None:
    training_loop = AccelerateTrainingLoop(clip_grad={"name": "clip_grad_norm"})
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (1, 2)


@mark.parametrize("max_norm", (0.1, 1))
@mark.parametrize("norm_type", (1, 2))
def test_accelerate_training_loop_clip_grad_clip_grad_norm_with_max_norm_and_norm_type(
    max_norm: float, norm_type: float
) -> None:
    training_loop = AccelerateTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": max_norm, "norm_type": norm_type}
    )
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (max_norm, norm_type)


def test_accelerate_training_loop_clip_grad_incorrect_name() -> None:
    with raises(ValueError, match=r"Incorrect clip grad name \(incorrect name\)."):
        AccelerateTrainingLoop(clip_grad={"name": "incorrect name"})


def test_accelerate_training_loop_observer_default() -> None:
    assert isinstance(AccelerateTrainingLoop()._observer, NoOpLoopObserver)


def test_accelerate_training_loop_observer(tmp_path: Path) -> None:
    assert isinstance(
        AccelerateTrainingLoop(observer=PyTorchBatchSaver(tmp_path))._observer,
        PyTorchBatchSaver,
    )


def test_accelerate_training_loop_no_profiler() -> None:
    assert isinstance(AccelerateTrainingLoop()._profiler, NoOpProfiler)


def test_accelerate_training_loop_profiler_tensorboard() -> None:
    assert isinstance(
        AccelerateTrainingLoop(profiler=PyTorchProfiler(torch.profiler.profile()))._profiler,
        PyTorchProfiler,
    )


def test_accelerate_training_loop_train() -> None:
    engine = create_dummy_engine()
    AccelerateTrainingLoop().train(engine)
    assert engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == 3
    loss_history = engine.get_history(f"train/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 1


def test_accelerate_training_loop_train_loss_nan() -> None:
    engine = create_dummy_engine(model=DummyClassificationModel(loss_nan=True))
    AccelerateTrainingLoop().train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    with raises(EmptyHistoryError, match=f"'train/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"train/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


def test_accelerate_training_loop_train_with_loss_history() -> None:
    engine = create_dummy_engine()
    engine.add_history(MinScalarHistory(f"train/{ct.LOSS}"))
    engine.log_metric(f"train/{ct.LOSS}", 1, EpochStep(-1))
    AccelerateTrainingLoop().train(engine)
    loss_history = engine.get_history(f"train/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 2


def test_accelerate_training_loop_train_set_grad_to_none_true() -> None:
    engine = create_dummy_engine()
    AccelerateTrainingLoop(set_grad_to_none=True).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    loss_history = engine.get_history(f"train/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)


def test_accelerate_training_loop_train_with_clip_grad_value() -> None:
    engine = create_dummy_engine()
    AccelerateTrainingLoop(clip_grad={"name": "clip_grad_value", "clip_value": 0.25}).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    assert isinstance(engine.get_history(f"train/{ct.LOSS}").get_last_value(), float)


def test_accelerate_training_loop_train_with_clip_grad_norm() -> None:
    engine = create_dummy_engine()
    AccelerateTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2}
    ).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    assert isinstance(engine.get_history(f"train/{ct.LOSS}").get_last_value(), float)


# TODO: Comment this test because the current version of accelerate does not support
#  empty data loader
#
# def test_accelerate_training_loop_train_empty_map_dataset() -> None:
#     engine = create_dummy_engine(datasource=FakeDataSource(train_dataset=EmptyFakeMapDataset()))
#     AccelerateTrainingLoop().train(engine)
#     assert engine.epoch == -1
#     assert engine.iteration == -1
#     with raises(EmptyHistoryError, match=f"'train/{ct.LOSS}' history is empty."):
#         # The loss is not logged because there is no batch
#         engine.get_history(f"train/{ct.LOSS}").get_last_value()


def test_accelerate_training_loop_train_iterable_dataset() -> None:
    engine = create_dummy_engine(
        datasource=DummyDataSource(train_dataset=DummyIterableDataset(), batch_size=1)
    )
    AccelerateTrainingLoop().train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 7
    assert isinstance(engine.get_history(f"train/{ct.LOSS}").get_last_value(), float)


# TODO: Comment this test because the current version of accelerate does not support
#  empty data loader
#
# def test_accelerate_training_loop_train_empty_iterable_dataset() -> None:
#     engine = create_dummy_engine(
#         datasource=FakeDataSource(train_dataset=EmptyFakeIterableDataset(), batch_size=None)
#     )
#     AccelerateTrainingLoop().train(engine)
#     assert engine.epoch == -1
#     assert engine.iteration == -1
#     with raises(EmptyHistoryError, match=f"'train/{ct.LOSS}' history is empty."):
#         # The loss is not logged because there is no batch
#         engine.get_history(f"train/{ct.LOSS}").get_last_value()


@mark.parametrize("event", (EngineEvents.TRAIN_EPOCH_STARTED, EngineEvents.TRAIN_EPOCH_COMPLETED))
def test_accelerate_training_loop_fire_event_train_epoch_events(event: str) -> None:
    engine = create_dummy_engine()
    engine.add_event_handler(
        event, GEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    engine.increment_epoch()  # simulate epoch 0
    AccelerateTrainingLoop().train(engine)
    assert engine.epoch == 2
    assert engine.iteration == 3


@mark.parametrize(
    "event",
    (
        EngineEvents.TRAIN_ITERATION_STARTED,
        EngineEvents.TRAIN_FORWARD_COMPLETED,
        EngineEvents.TRAIN_BACKWARD_COMPLETED,
        EngineEvents.TRAIN_ITERATION_COMPLETED,
    ),
)
def test_accelerate_training_loop_train_fire_event_train_iteration_events(event: str) -> None:
    engine = create_dummy_engine()
    engine.add_event_handler(
        event, GEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    engine.increment_epoch()  # simulate epoch 0
    AccelerateTrainingLoop().train(engine)
    assert engine.epoch == 8
    assert engine.iteration == 3


def test_accelerate_training_loop_train_with_observer() -> None:
    engine = create_dummy_engine()
    observer = Mock(spec=BaseLoopObserver)
    AccelerateTrainingLoop(observer=observer).train(engine)
    observer.start.assert_called_once_with(engine)
    assert observer.update.call_count == 4
    observer.end.assert_called_once_with(engine)


def test_accelerate_training_loop_train_with_profiler() -> None:
    profiler = MagicMock(spec=BaseProfiler)
    AccelerateTrainingLoop(profiler=profiler).train(engine=create_dummy_engine())
    assert profiler.__enter__().step.call_count == 4


def test_accelerate_training_loop_load_state_dict() -> None:
    AccelerateTrainingLoop().load_state_dict({})  # Verify it does not raise error


def test_accelerate_training_loop_state_dict() -> None:
    assert AccelerateTrainingLoop().state_dict() == {}


def test_vanilla_training_loop_train_one_batch_fired_events() -> None:
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel()
    AccelerateTrainingLoop()._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert engine.fire_event.call_args_list == [
        ((EngineEvents.TRAIN_ITERATION_STARTED,), {}),
        ((EngineEvents.TRAIN_FORWARD_COMPLETED,), {}),
        ((EngineEvents.TRAIN_BACKWARD_COMPLETED,), {}),
        ((EngineEvents.TRAIN_ITERATION_COMPLETED,), {}),
    ]


@mark.parametrize("set_grad_to_none", (True, False))
def test_vanilla_training_loop_train_one_batch_set_grad_to_none(set_grad_to_none: bool) -> None:
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel()
    out = AccelerateTrainingLoop(
        set_grad_to_none=set_grad_to_none,
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])


def test_vanilla_training_loop_train_one_batch_clip_grad_value() -> None:
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel()
    out = AccelerateTrainingLoop(
        clip_grad={"name": "clip_grad_value", "clip_value": 0.25},
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])


def test_vanilla_training_loop_train_one_batch_clip_grad_norm() -> None:
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel()
    out = AccelerateTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2},
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])


def test_vanilla_training_loop_train_one_batch_loss_nan() -> None:
    engine = Mock(spec=BaseEngine)
    model = Mock(spec=Module, return_value={ct.LOSS: torch.tensor(float("nan"))})
    optimizer = Mock(spec=Optimizer)
    out = AccelerateTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2}
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=optimizer,
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.isnan(out[ct.LOSS])
    assert engine.fire_event.call_args_list == [
        ((EngineEvents.TRAIN_ITERATION_STARTED,), {}),
        ((EngineEvents.TRAIN_FORWARD_COMPLETED,), {}),
        ((EngineEvents.TRAIN_ITERATION_COMPLETED,), {}),
    ]
