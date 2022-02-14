from copy import deepcopy

from ray.rllib.agents.dqn.simple_q import SimpleQTrainer
from ray.rllib.utils.annotations import override

from value_mix.value_mix_execution_plan import value_mix_execution_plan
from value_mix.policy import ValueMixPolicy
from value_mix.configs.default_config import DEFAULT_CONFIG



class ValueMixTrainer(SimpleQTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls):
        return DEFAULT_CONFIG

    @override(SimpleQTrainer)
    def validate_config(self, config):
        # Call super's validation method.
        super().validate_config(config)

        if config["framework"] != "torch":
            raise ValueError("Only `framework=torch` supported so far for ValueMixTrainer!")

    @override(SimpleQTrainer)
    def get_default_policy_class(self, config):
        return ValueMixPolicy

    @staticmethod
    @override(SimpleQTrainer)
    def execution_plan(workers, config, **kwargs):
        assert (
            len(kwargs) == 0
        ), "ValueMix execution_plan does NOT take any additional parameters"

        return value_mix_execution_plan(workers, config)



vdn_config = deepcopy(DEFAULT_CONFIG)
vdn_config.update({"mixer": "vdn"})
class VDNTrainer(ValueMixTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls):
        return vdn_config


qmix_config = deepcopy(DEFAULT_CONFIG)
qmix_config.update({"mixer": "qmix"})
class QMixTrainer(ValueMixTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls):
        return qmix_config


qplex_config = deepcopy(DEFAULT_CONFIG)
qplex_config.update({"mixer": "qplex"})
class QplexTrainer(ValueMixTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls):
        return qplex_config

qatten_config = deepcopy(DEFAULT_CONFIG)
qatten_config.update({"mixer": "qatten"})
class QattenTrainer(ValueMixTrainer):
    @classmethod
    @override(SimpleQTrainer)
    def get_default_config(cls):
        return qatten_config

