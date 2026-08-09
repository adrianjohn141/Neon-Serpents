from __future__ import annotations

import importlib
import queue
import sys
import threading
from concurrent import futures
from pathlib import Path

import grpc

generated = Path(__file__).parents[1] / "generated"
if str(generated) not in sys.path:
    sys.path.insert(0, str(generated))
training_pb2 = importlib.import_module("training_pb2")
training_pb2_grpc = importlib.import_module("training_pb2_grpc")


class ActorBridge(training_pb2_grpc.TrainingActorsServicer):
    def __init__(self) -> None:
        self.incoming: queue.Queue = queue.Queue(maxsize=16)
        self.outgoing: queue.Queue = queue.Queue()
        self.connected = threading.Event()
        self.current_start = None

    def Connect(self, request_iterator, context):  # noqa: N802
        reader_done = threading.Event()

        def read() -> None:
            try:
                for request in request_iterator:
                    if request.HasField("hello"):
                        self.connected.set()
                        if self.current_start is not None:
                            self.outgoing.put(training_pb2.LearnerToActor(start=self.current_start))
                    elif request.HasField("batch"):
                        self.incoming.put(request.batch)
            except grpc.RpcError:
                # A disappearing actor stream is a normal lifecycle event. The
                # coordinator reconnects when the next learner session opens.
                pass
            finally:
                reader_done.set()

        threading.Thread(target=read, daemon=True).start()
        while context.is_active() and not reader_done.is_set():
            try:
                yield self.outgoing.get(timeout=1)
            except queue.Empty:
                continue

    def start_job(
        self,
        experiment_id: str,
        actor_count: int,
        budget: int,
        seed: int,
        policies: list[dict],
        training_spec_version: int = 3,
        observation_size: int = 228,
        roster: list[dict] | None = None,
        battle_size: int = 4,
    ) -> None:
        converted = [self.policy_message(policy) for policy in policies]
        self.current_start = training_pb2.StartJob(
            experiment_id=experiment_id,
            actor_count=actor_count,
            step_budget=budget,
            seed=seed,
            policies=converted,
            training_spec_version=training_spec_version,
            observation_size=observation_size,
            roster=[training_pb2.SnakeRosterEntry(
                snake_id=entry["id"], name=entry["name"], color=entry["color"], accent=entry["accent"]
            ) for entry in (roster or [])],
            battle_size=battle_size,
        )
        if self.connected.is_set():
            self.outgoing.put(training_pb2.LearnerToActor(start=self.current_start))

    def policy_message(self, policy: dict):
        return training_pb2.PolicyWeights(
            snake_id=policy["snakeId"], version=policy["version"], epsilon=policy["epsilon"],
            environment_steps=policy["environmentSteps"],
            survival_steps=policy["scenarioSteps"]["survival"],
            powerup_steps=policy["scenarioSteps"]["powerup"],
            battle_steps=policy["scenarioSteps"]["battle"],
            safe_zone_steps=policy["scenarioSteps"]["safezone"],
            hazard_steps=policy["scenarioSteps"]["hazard"],
            objective_steps=policy["scenarioSteps"]["objective"],
            series_steps=policy["scenarioSteps"]["series"],
            tensors=[training_pb2.TensorData(name=tensor["name"], shape=tensor["shape"], values=tensor["values"]) for tensor in policy["tensors"]],
        )

    def publish_policies(self, policies: list[dict]) -> None:
        for policy in policies:
            self.outgoing.put(training_pb2.LearnerToActor(policy=self.policy_message(policy)))

    def acknowledge(self, sequence: int) -> None:
        self.outgoing.put(training_pb2.LearnerToActor(ack=training_pb2.BatchAck(sequence=sequence)))

    def stop_job(self, reason: str) -> None:
        self.outgoing.put(training_pb2.LearnerToActor(stop=training_pb2.StopJob(reason=reason)))
        self.current_start = None


def serve(bind: str) -> tuple[grpc.Server, ActorBridge]:
    bridge = ActorBridge()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)])
    training_pb2_grpc.add_TrainingActorsServicer_to_server(bridge, server)
    server.add_insecure_port(bind)
    server.start()
    return server, bridge
