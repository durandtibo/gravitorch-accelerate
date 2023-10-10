from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.loops.evaluation.conditions import (
    EveryEpochEvalCondition,
    LastEpochEvalCondition,
)
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
from objectory import OBJECT_TARGET
from pytest import fixture, mark, raises

from gtaccelerate.loops.evaluation import AccelerateEvaluationLoop

##############################################
#     Tests for AccelerateEvaluationLoop     #
##############################################


@fixture(autouse=True)
def reset_accelerate_state() -> None:
    AcceleratorState._reset_state(reset_partial_state=True)


def increment_epoch_handler(engine: BaseEngine) -> None:
    engine.increment_epoch(2)


def test_accelerate_evaluation_loop_str() -> None:
    assert str(AccelerateEvaluationLoop()).startswith("AccelerateEvaluationLoop(")


def test_accelerate_evaluation_loop_accelerator_none() -> None:
    assert isinstance(AccelerateEvaluationLoop()._accelerator, Accelerator)


def test_accelerate_evaluation_loop_accelerator_object() -> None:
    evaluation_loop = AccelerateEvaluationLoop(accelerator=Accelerator(cpu=True))
    assert isinstance(evaluation_loop._accelerator, Accelerator)
    assert evaluation_loop._accelerator.state.device.type == "cpu"


def test_accelerate_evaluation_loop_accelerator_from_dict() -> None:
    assert (
        AccelerateEvaluationLoop(accelerator={"cpu": True})._accelerator.state.device.type == "cpu"
    )


@mark.parametrize("tag", ("val", "test"))
def test_accelerate_evaluation_loop_tag(tag: str) -> None:
    assert AccelerateEvaluationLoop(tag=tag)._tag == tag


def test_accelerate_evaluation_loop_tag_default() -> None:
    assert AccelerateEvaluationLoop()._tag == "eval"


def test_accelerate_evaluation_loop_condition() -> None:
    evaluation_loop = AccelerateEvaluationLoop(
        condition={OBJECT_TARGET: "gravitorch.loops.evaluation.conditions.LastEpochEvalCondition"}
    )
    assert isinstance(evaluation_loop._condition, LastEpochEvalCondition)


def test_accelerate_evaluation_loop_condition_default() -> None:
    assert isinstance(AccelerateEvaluationLoop()._condition, EveryEpochEvalCondition)


def test_accelerate_evaluation_loop_observer(tmp_path: Path) -> None:
    assert isinstance(
        AccelerateEvaluationLoop(observer=PyTorchBatchSaver(tmp_path))._observer,
        PyTorchBatchSaver,
    )


def test_accelerate_evaluation_loop_observer_default() -> None:
    assert isinstance(AccelerateEvaluationLoop()._observer, NoOpLoopObserver)


def test_accelerate_evaluation_loop_no_profiler() -> None:
    assert isinstance(AccelerateEvaluationLoop()._profiler, NoOpProfiler)


def test_accelerate_evaluation_loop_profiler_tensorboard() -> None:
    assert isinstance(
        AccelerateEvaluationLoop(profiler=PyTorchProfiler(torch.profiler.profile()))._profiler,
        PyTorchProfiler,
    )


def test_accelerate_evaluation_loop_eval() -> None:
    engine = create_dummy_engine()
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert not engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == -1
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 1


def test_accelerate_evaluation_loop_eval_loss_nan() -> None:
    engine = create_dummy_engine(model=DummyClassificationModel(loss_nan=True))
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


def test_accelerate_evaluation_loop_eval_with_loss_history() -> None:
    engine = create_dummy_engine()
    engine.add_history(MinScalarHistory(f"eval/{ct.LOSS}"))
    engine.log_metric(f"eval/{ct.LOSS}", 1, EpochStep(-1))
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 2


def test_accelerate_evaluation_loop_eval_no_dataset() -> None:
    engine = create_dummy_engine(datasource=Mock(has_dataloader=Mock(return_value=False)))
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


# TODO: Comment this test because the current version of accelerate does not support
#  empty data loader
#
# def test_accelerate_evaluation_loop_eval_empty_map_dataset()-> None:
#     engine = create_dummy_engine(datasource=FakeDataSource(eval_dataset=EmptyFakeMapDataset()))
#     evaluation_loop = AccelerateEvaluationLoop(accelerator={'dispatch_batches': False})
#     evaluation_loop.eval(engine)
#     assert engine.epoch == -1
#     assert engine.iteration == -1
#     with raises(EmptyHistoryError,match=f"'eval/{ct.LOSS}' history is empty."):
#         # The loss is not logged because there is no batch
#         engine.get_history(f"eval/{ct.LOSS}").get_last_value()


def test_accelerate_evaluation_loop_eval_iterable_dataset() -> None:
    engine = create_dummy_engine(
        datasource=DummyDataSource(eval_dataset=DummyIterableDataset(), batch_size=1)
    )
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    assert isinstance(engine.get_history(f"eval/{ct.LOSS}").get_last_value(), float)


# TODO: Comment this test because the current version of accelerate does not support empty
#  data loader
#
# def test_accelerate_evaluation_loop_eval_empty_iterable_dataset()-> None:
#     engine = create_dummy_engine(
#         datasource=FakeDataSource(eval_dataset=EmptyFakeIterableDataset(), batch_size=None)
#     )
#     evaluation_loop = AccelerateEvaluationLoop(accelerator={"dispatch_batches": False})
#     evaluation_loop.eval(engine)
#     assert engine.epoch == -1
#     assert engine.iteration == -1
#     with raises(EmptyHistoryError,match=f"'eval/{ct.LOSS}' history is empty."):
#         # The loss is not logged because there is no batch
#         engine.get_history(f"eval/{ct.LOSS}").get_last_value()


def test_accelerate_evaluation_loop_eval_skip_evaluation() -> None:
    engine = create_dummy_engine()
    evaluation_loop = AccelerateEvaluationLoop(condition=Mock(return_value=False))
    evaluation_loop.eval(engine)
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@mark.parametrize("event", (EngineEvents.EVAL_EPOCH_STARTED, EngineEvents.EVAL_EPOCH_COMPLETED))
def test_accelerate_evaluation_loop_fire_event_eval_epoch_events(event: str) -> None:
    engine = create_dummy_engine()
    engine.add_event_handler(
        event, GEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == 1
    assert engine.iteration == -1


@mark.parametrize(
    "event", (EngineEvents.EVAL_ITERATION_STARTED, EngineEvents.EVAL_ITERATION_COMPLETED)
)
def test_accelerate_evaluation_loop_fire_event_eval_iteration_events(event: str) -> None:
    engine = create_dummy_engine()
    engine.add_event_handler(
        event, GEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == 7
    assert engine.iteration == -1


def test_accelerate_evaluation_loop_grad_enabled_false() -> None:
    engine = create_dummy_engine()
    loop = AccelerateEvaluationLoop(grad_enabled=False)
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = loop._eval_one_batch(engine, engine.model, batch)
    assert not out[ct.LOSS].requires_grad


def test_accelerate_evaluation_loop_grad_enabled_true() -> None:
    engine = create_dummy_engine()
    loop = AccelerateEvaluationLoop(grad_enabled=True)
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = loop._eval_one_batch(engine, engine.model, batch)
    assert out[ct.LOSS].requires_grad
    out[ct.LOSS].backward()
    assert torch.is_tensor(batch[ct.INPUT].grad)


def test_accelerate_evaluation_loop_train_with_observer() -> None:
    engine = create_dummy_engine()
    observer = Mock(spec=BaseLoopObserver)
    AccelerateEvaluationLoop(observer=observer).eval(engine)
    observer.start.assert_called_once_with(engine)
    assert observer.update.call_count == 4
    observer.end.assert_called_once_with(engine)


def test_accelerate_evaluation_loop_eval_with_profiler() -> None:
    profiler = MagicMock(spec=BaseProfiler)
    training_loop = AccelerateEvaluationLoop(profiler=profiler)
    training_loop.eval(engine=create_dummy_engine())
    assert profiler.__enter__().step.call_count == 4


def test_accelerate_evaluation_loop_load_state_dict() -> None:
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.load_state_dict({})


def test_accelerate_evaluation_loop_state_dict() -> None:
    assert AccelerateEvaluationLoop().state_dict() == {}
