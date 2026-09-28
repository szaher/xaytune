"""Commit reports become durable observations, with no automatic resume."""

from __future__ import annotations

import asyncio

from tests.test_checkpoints.helpers import make_bundle
from tests.test_experiment.test_incidents import _ReplayRuntime, _seed
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, RuntimeRef
from xaytune.experiment import EmbeddedControllerHost
from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload
from xaytune.storage import write_transaction


def test_controller_restart_records_redelivered_checkpoint_once_without_resume(tmp_path):
    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        experiment, run, attempt, owner = _seed(host)
        state, context, restore = make_bundle(
            tmp_path / "source",
            attempt_id=attempt.id,
            candidate=run.candidate_fingerprint,
        )
        attempt = type(attempt).model_validate(
            {
                **attempt.model_dump(mode="json"),
                "execution_fingerprint": context.execution_fingerprint,
            }
        )
        with write_transaction(host.repository._connection):
            host.repository._connection.execute(
                "UPDATE run_attempts SET payload_json = ? WHERE id = ?",
                (attempt.model_dump_json(), str(attempt.id)),
            )
        store = LocalCheckpointStore(tmp_path / "bundles")
        manager = CheckpointManager(SerializedStateCodec(), store)
        ref = await manager.save(state, context)
        manifest = (await store.get(ref)).manifest
        event = RuntimeEventEnvelope(
            event_id="checkpoint-commit",
            sequence=3,
            target=owner.target,
            payload=TrainingEventPayload(data=manifest.committed_payload(ref)),
        )
        original = host.repository.record_checkpoint(
            attempt.id,
            event.payload.data,
            evidence=FrozenDict(event.model_dump(mode="json")),
            actor=Actor(type="system", id="controller"),
        )
        await host.close()
        restarted = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            await restarted._observe(
                experiment.id,
                attempt.id,
                run.id,
                _ReplayRuntime(event),
                RuntimeRef(backend="local", external_id="failed-workload"),
            )
            assert restarted.repository.checkpoints.for_attempt(str(attempt.id)) == (original,)
            assert (await manager.restore_recorded(original, restore)).manifest == manifest
            connection = restarted.repository._connection
            for table, count in (
                ("run_attempts", 1),
                ("experiment_nodes", 1),
                ("actions", 0),
                ("runtime_operations", 0),
                ("incidents", 0),
            ):
                assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count
            events = restarted.repository.events.events_for_aggregate(str(attempt.id))
            assert sum(e.event_type == "CheckpointRecorded" for e in events) == 1
        finally:
            await restarted.close()

    asyncio.run(scenario())
