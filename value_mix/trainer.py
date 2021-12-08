from copy import deepcopy

from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer

from value_mix.value_mix_execution_plan import value_mix_execution_plan
from value_mix.policy import ValueMixPolicy
from value_mix.configs.default_config import DEFAULT_CONFIG

vdn_config = deepcopy(DEFAULT_CONFIG)
vdn_config.update({"mixer": "vdn"})
VDNTrainer = GenericOffPolicyTrainer.with_updates(
    name="VDN",
    default_config=vdn_config,
    default_policy=ValueMixPolicy,
    get_policy_class=None,
    execution_plan=value_mix_execution_plan)

qmix_config = deepcopy(DEFAULT_CONFIG)
qmix_config.update({"mixer": "qmix"})
QMixTrainer = GenericOffPolicyTrainer.with_updates(
    name="QMIX",
    default_config=qmix_config,
    default_policy=ValueMixPolicy,
    get_policy_class=None,
    execution_plan=value_mix_execution_plan)


qplex_config = deepcopy(DEFAULT_CONFIG)
qplex_config.update({"mixer": "qplex"})
QplexTrainer = GenericOffPolicyTrainer.with_updates(
    name="QPLEX",
    default_config=qplex_config,
    default_policy=ValueMixPolicy,
    get_policy_class=None,
    execution_plan=value_mix_execution_plan)

qatten_config = deepcopy(DEFAULT_CONFIG)
qatten_config.update({"mixer": "qatten"})
QattenTrainer = GenericOffPolicyTrainer.with_updates(
    name="QATTEN",
    default_config=qatten_config,
    default_policy=ValueMixPolicy,
    get_policy_class=None,
    execution_plan=value_mix_execution_plan)

