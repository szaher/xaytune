"""Native single-worker FULL+EXACT checkpoint capture and application.

The checkpoint manager owns byte integrity and publication. This adapter owns
PyTorch state and the indexed sample stream. It deliberately refuses streaming,
distributed and multiprocessing loaders until they have an exact cursor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import tempfile
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from xaytune.checkpoints import (
    CheckpointManager,
    CheckpointState,
    LocalCheckpointStore,
    SerializedStateCodec,
)
from xaytune.checkpoints._files import describe
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.checkpoint import CheckpointCompatibilityKey, CheckpointContext, RestoreContext
from xaytune.core.clock import utc_now
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ArtifactId, CheckpointId, RunAttemptId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import ArtifactRef, CheckpointRef
from xaytune.core.resume import (
    CheckpointBoundary,
    CheckpointStateManifest,
    DataCursor,
    DataResume,
    ResumeGuarantee,
    RNGState,
    SamplerState,
    StateRestore,
    WorkerRNGState,
)
from xaytune.runtimes.worker import ObservationWriter
from xaytune.trainer.callbacks import CallbackManager, TrainState

_GUARANTEE = ResumeGuarantee(
    state=StateRestore.FULL, data=DataResume.EXACT, boundary=CheckpointBoundary.OPTIMIZER_STEP
)


class ExactIndexedSampler(Sampler[int]):
    """Epoch-seeded permutation with a sample cursor independent of batch size."""

    def __init__(self, size: int, seed: int, *, epoch: int = 0, offset: int = 0) -> None:
        if size < 1 or not 0 <= offset <= size or epoch < 0:
            raise ValueError("invalid exact indexed sampler position")
        self.size = size
        self.seed = seed
        self.epoch = epoch
        self.offset = offset

    def set_epoch(self, epoch: int, offset: int = 0) -> None:
        if epoch < 0 or not 0 <= offset <= self.size:
            raise ValueError("invalid exact indexed sampler position")
        self.epoch, self.offset = epoch, offset

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()[self.offset :]

    def __len__(self) -> int:
        return self.size - self.offset


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _model_input_digest(directory: Path) -> str:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Native managed checkpoint requires a stable local model directory")
    entries = sorted(directory.rglob("*"))
    files = [path for path in entries if path.is_file()]
    if not files or any(path.is_symlink() for path in entries):
        raise ValueError("Native model input is missing or contains symlinks")
    return fingerprint(
        tuple((path.relative_to(directory).as_posix(), _digest(path)) for path in files)
    )


def _reference(source: Path, name: str, attempt_id: RunAttemptId) -> ArtifactRef:
    return ArtifactRef(
        id=ArtifactId.generate(),
        kind="checkpoint_state",
        uri=name,
        digest=describe(source / name, name).digest,
        producer_attempt_id=attempt_id,
    )


def _tuples(value: Any) -> Any:
    return tuple(_tuples(part) for part in value) if isinstance(value, list) else value


def native_restore_context(
    plan: ResolvedExecutionPlan, data_path: Path, *, seed: int
) -> RestoreContext:
    """Construct the intended Native consumer identity without reading a checkpoint.

    Batch size is deliberately absent: an OOM successor changes it while
    retaining the same scientific candidate, input bytes and sample order.
    """
    if not isinstance(plan.spec, TrainingExecutionSpec):
        raise ValueError("Native restore context requires a training plan")
    if plan.spec.checkpoint.format != "native-torch/v1" or not data_path.is_file():
        raise ValueError("Native restore context requires managed format and local data")
    model_input_digest = _model_input_digest(Path(plan.spec.config["model"]["uri"]))
    dataset_fingerprint = fingerprint(
        {
            "path_digest": _digest(data_path),
            "data": plan.spec.config["data"],
            "model_input_digest": model_input_digest,
        }
    )
    return RestoreContext(
        candidate_fingerprint=plan.spec.candidate_fingerprint,
        compatibility=CheckpointCompatibilityKey(
            state_format="native-torch/v1",
            model_fingerprint=fingerprint(
                {"candidate": plan.spec.candidate_fingerprint, "input": model_input_digest}
            ),
            optimizer_layout="torch-adamw/v1",
            scheduler_layout=f"torch-{plan.spec.config['optimization']['scheduler']}/v1",
            distributed_strategy="single-process",
            sharding_scheme="none",
            topology_fingerprint=f"single-worker:cuda={torch.cuda.device_count()}",
            framework_versions=FrozenDict(
                {
                    "torch": torch.__version__.split("+")[0],
                    "numpy": np.__version__,
                    "transformers": version("transformers"),
                }
            ),
        ),
        dataset_fingerprint=dataset_fingerprint,
        ordering_fingerprint=fingerprint(
            {"sampler": "native-indexed/v1", "dataset": dataset_fingerprint, "seed": seed}
        ),
        required_guarantee=_GUARANTEE,
    )


class NativeCheckpointAdapter:
    """Apply and capture the complete supported Native trainer state."""

    def __init__(
        self,
        plan: ResolvedExecutionPlan,
        data_path: Path,
        *,
        seed: int,
        dataset_size: int,
        micro_batch_size: int,
        gradient_accumulation: int,
        writer: ObservationWriter,
    ) -> None:
        if (
            not isinstance(plan.spec, TrainingExecutionSpec)
            or plan.target.kind != "training-attempt"
        ):
            raise ValueError("Native checkpoint adapter requires a training attempt")
        store_uri = plan.spec.checkpoint.store_uri
        if store_uri is None or plan.spec.checkpoint.format != "native-torch/v1":
            raise ValueError("Native managed checkpoint requires its declared store and format")
        if not data_path.is_file():
            raise ValueError("Native exact resume requires a local indexed dataset")
        if dataset_size < 1:
            raise ValueError("Native exact resume requires a nonempty dataset")
        effective_batch = micro_batch_size * gradient_accumulation
        if effective_batch < 1 or dataset_size % effective_batch:
            raise ValueError("Native exact resume requires full optimizer windows in every epoch")
        self.plan = plan
        self.spec = plan.spec
        self.attempt_id = RunAttemptId.validate(plan.target.id)
        self.manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(Path(store_uri))
        )
        self.writer = writer
        self.seed = seed
        self.dataset_size = dataset_size
        self.effective_batch = effective_batch
        context = native_restore_context(plan, data_path, seed=seed)
        assert context.dataset_fingerprint is not None
        assert context.ordering_fingerprint is not None
        self.dataset_fingerprint = context.dataset_fingerprint
        self.ordering_fingerprint = context.ordering_fingerprint
        self.compatibility = context.compatibility
        self.sampler = ExactIndexedSampler(dataset_size, seed)
        self._initial_epoch = 0
        self._initial_offset = 0
        self._current_base_offset = 0

    def loader(self, original: DataLoader[Any], batch_size: int) -> DataLoader[Any]:
        if (
            original.num_workers != 0
            or original.batch_size != batch_size
            or not isinstance(original.dataset, list)
        ):
            raise ValueError("Native exact resume requires a single-process indexed loader")
        if len(cast(Any, original.dataset)) != self.dataset_size:
            raise ValueError("dataset changed before managed checkpoint setup")
        return DataLoader(
            original.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=original.collate_fn,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def restore_context(self) -> RestoreContext:
        return native_restore_context(
            self.plan, Path(self.spec.config["data"]["path"]), seed=self.seed
        )

    def bind_restore(self, reference: CheckpointRef) -> Any:
        """Validate/materialize now; apply after Trainer constructs its optimizer."""
        restored = asyncio.run(self.manager.restore(reference, self.restore_context()))
        cursor = restored.manifest.data_cursor
        if cursor is None or cursor.epoch is None or cursor.next_sample_offset is None:
            raise ValueError("Native exact checkpoint lacks an indexed cursor")
        sampler_ref = cursor.sampler_state
        if (
            sampler_ref is None
            or sampler_ref.provider != "native-indexed"
            or sampler_ref.version != "1"
        ):
            raise ValueError("Native exact checkpoint lacks sampler state")
        capture = json.loads((restored.source / sampler_ref.state_ref.uri).read_text())
        if (
            capture
            != {
                "provider": "native-indexed/v1",
                "seed": self.seed,
                "epoch": cursor.epoch,
                "offset": cursor.next_sample_offset,
                "size": self.dataset_size,
            }
            or cursor.next_sample_offset % self.effective_batch
        ):
            raise ValueError("Native checkpoint cursor disagrees with its sampler capture")
        self._initial_epoch = cursor.epoch
        self._initial_offset = cursor.next_sample_offset
        self.sampler.set_epoch(self._initial_epoch, self._initial_offset)

        def apply(model: Any, optimizer: Any, scheduler: Any, scaler: Any) -> TrainState:
            state = restored.manifest.state_manifest
            if (
                optimizer is None
                or scheduler is None
                or state.optimizer is None
                or state.scheduler is None
            ):
                raise ValueError("Native FULL restore requires optimizer and scheduler")
            source = restored.source
            model.load_state_dict(
                torch.load(source / state.model.uri, weights_only=True, map_location="cpu")
            )
            scheduler.load_state_dict(
                torch.load(source / state.scheduler.uri, weights_only=True, map_location="cpu")
            )
            optimizer.load_state_dict(
                torch.load(source / state.optimizer.uri, weights_only=True, map_location="cpu")
            )
            if state.scaler is None or state.rng is None:
                raise ValueError("Native FULL restore requires scaler and RNG capture")
            scaler_capture = json.loads((source / state.scaler.uri).read_text())
            if scaler_capture["applicable"] != (scaler is not None):
                raise ValueError("checkpoint scaler applicability changed")
            if scaler is not None:
                scaler.load_state_dict(
                    torch.load(source / "scaler.pt", weights_only=True, map_location="cpu")
                )
            rng = state.rng
            random.setstate(_tuples(json.loads((source / rng.python.uri).read_text())))
            numpy_state = json.loads((source / rng.numpy.uri).read_text())
            np.random.set_state(
                (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
            )
            torch.set_rng_state(torch.load(source / rng.torch_cpu.uri, weights_only=True))
            if rng.workers:
                accelerator = rng.workers[0].accelerator
                if accelerator is None or not torch.cuda.is_available():
                    raise ValueError("checkpoint accelerator RNG cannot be restored")
                accelerator_states = torch.load(
                    source / accelerator.uri, weights_only=True, map_location="cpu"
                )
                if len(accelerator_states) != torch.cuda.device_count():
                    raise ValueError("checkpoint accelerator topology changed")
                torch.cuda.set_rng_state_all(accelerator_states)
            elif torch.cuda.is_available():
                raise ValueError("checkpoint omitted available accelerator RNG")
            return TrainState(
                step=-1, epoch=cast(int, cursor.epoch), global_step=restored.manifest.optimizer_step
            )

        return apply

    def register_capture(
        self,
        callbacks: CallbackManager,
        *,
        trainer: Any,
        model: Any,
        batch_size: int,
        every_steps: int,
    ) -> None:
        @callbacks.on("epoch_start")
        def _epoch(state: TrainState) -> None:
            offset = self._initial_offset if state.epoch == self._initial_epoch else 0
            self.sampler.set_epoch(state.epoch, offset)
            self._current_base_offset = offset

        @callbacks.on("step_end")
        def _capture(state: TrainState) -> None:
            if not trainer._last_optimizer_step_applied:
                raise ValueError("Native exact recovery cannot skip an optimizer update")
            if every_steps <= 0 or state.global_step % every_steps:
                return
            offset = min(
                self.dataset_size, self._current_base_offset + (state.step + 1) * batch_size
            )
            self.capture(trainer, model, state, offset)

    def capture(self, trainer: Any, model: Any, state: TrainState, offset: int) -> CheckpointRef:
        if not 0 <= offset <= self.dataset_size or offset % self.effective_batch:
            raise ValueError("Native exact checkpoint requires a consumed optimizer window")
        optimizer, scheduler, scaler = trainer._optimizer, trainer._scheduler, trainer._scaler
        if optimizer is None or scheduler is None or trainer._is_ds:
            raise ValueError("Native FULL checkpoint requires trainer-owned optimizer/scheduler")
        if trainer._accum_count % trainer.config.gradient_accumulation:
            raise ValueError("Native FULL checkpoint requires an optimizer boundary")
        with tempfile.TemporaryDirectory(prefix="xaytune-native-state-") as temporary:
            source = Path(temporary)
            torch.save(model.state_dict(), source / "model.pt")
            torch.save(optimizer.state_dict(), source / "optimizer.pt")
            torch.save(scheduler.state_dict(), source / "scheduler.pt")
            (source / "scaler.json").write_text(json.dumps({"applicable": scaler is not None}))
            if scaler is not None:
                torch.save(scaler.state_dict(), source / "scaler.pt")
            (source / "python-rng.json").write_text(json.dumps(random.getstate()))
            numpy_state: Any = np.random.get_state()
            (source / "numpy-rng.json").write_text(
                json.dumps([numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])
            )
            torch.save(torch.get_rng_state(), source / "torch-rng.pt")
            workers: tuple[WorkerRNGState, ...] = ()
            if torch.cuda.is_available():
                torch.save(torch.cuda.get_rng_state_all(), source / "cuda-rng.pt")
                workers = (
                    WorkerRNGState(
                        logical_worker_id="rank0",
                        accelerator=_reference(source, "cuda-rng.pt", self.attempt_id),
                    ),
                )
            (source / "sampler.json").write_text(
                json.dumps(
                    {
                        "provider": "native-indexed/v1",
                        "seed": self.seed,
                        "epoch": state.epoch,
                        "offset": offset,
                        "size": self.dataset_size,
                    },
                    sort_keys=True,
                )
            )
            manifest = CheckpointStateManifest(
                model=_reference(source, "model.pt", self.attempt_id),
                optimizer=_reference(source, "optimizer.pt", self.attempt_id),
                scheduler=_reference(source, "scheduler.pt", self.attempt_id),
                scaler=_reference(source, "scaler.json", self.attempt_id),
                rng=RNGState(
                    python=_reference(source, "python-rng.json", self.attempt_id),
                    numpy=_reference(source, "numpy-rng.json", self.attempt_id),
                    torch_cpu=_reference(source, "torch-rng.pt", self.attempt_id),
                    workers=workers,
                ),
                micro_step=0,
                applied_intervention_application_ids=(),
            )
            cursor = DataCursor(
                dataset_fingerprint=self.dataset_fingerprint,
                ordering_fingerprint=self.ordering_fingerprint,
                epoch=state.epoch,
                next_sample_offset=offset,
                examples_seen=state.epoch * self.dataset_size + offset,
                sampler_state=SamplerState(
                    provider="native-indexed",
                    version="1",
                    state_ref=_reference(source, "sampler.json", self.attempt_id),
                ),
            )
            context = CheckpointContext(
                checkpoint_id=CheckpointId.generate(),
                producer_attempt_id=self.attempt_id,
                candidate_fingerprint=self.spec.candidate_fingerprint,
                execution_fingerprint=training_execution_fingerprint(self.plan),
                compatibility=self.compatibility,
                created_at=utc_now(),
            )
            reference = asyncio.run(
                self.manager.save(
                    CheckpointState(source, state.global_step, cursor, _GUARANTEE, manifest),
                    context,
                )
            )
        localized = asyncio.run(self.manager.store.get(reference))
        self.writer.write(localized.manifest.committed_payload(reference))
        return reference
