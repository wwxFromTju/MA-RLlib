from ray.rllib.execution.replay_ops import SimpleReplayBuffer, Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.concurrency_ops import Concurrently

def value_mix_execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")
    replay_buffer = SimpleReplayBuffer(config["buffer_size"])

    store_op = rollouts \
        .for_each(StoreToReplayBuffer(local_buffer=replay_buffer))

    train_op = Replay(local_buffer=replay_buffer) \
        .combine(
        ConcatBatches(
            min_batch_size=config["train_batch_size"],
            count_steps_by=config["multiagent"]["count_steps_by"]
        )) \
        .for_each(TrainOneStep(workers)) \
        .for_each(UpdateTargetNetwork(
            workers, config["target_network_update_freq"]))

    merged_op = Concurrently(
        [store_op, train_op], mode="round_robin", output_indexes=[1])

    return StandardMetricsReporting(merged_op, workers, config)
