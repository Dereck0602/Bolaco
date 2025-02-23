# License: MIT

from openbox import logger
from openbox.utils.config_space import ConfigurationSpace
from openbox.core.sync_batch_advisor import SyncBatchAdvisor, SUCCESS
from openbox.apps.multi_fidelity.mq_hb import mqHyperband
from openbox.apps.multi_fidelity.utils import sample_configurations, expand_configurations
from openbox.utils.history import Observation


class mqBOHB(mqHyperband):
    """ The implementation of BOHB.
        The paper can be found in https://arxiv.org/abs/1807.01774 .
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 num_iter=10000,
                 rand_prob=0.3,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqBOHB',
                 restart_needed=True,
                 max_runtime_per_trial=None,
                 max_runtime=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        super().__init__(objective_func, config_space, R, eta=eta, num_iter=num_iter,
                         random_state=random_state, method_id=method_id,
                         restart_needed=restart_needed, max_runtime_per_trial=max_runtime_per_trial,
                         max_runtime=max_runtime,
                         ip=ip, port=port, authkey=authkey)

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        # using median_imputation batch_strategy implemented in OpenBox to generate BO suggestions
        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        self.config_advisor = SyncBatchAdvisor(config_space,
                                               num_objectives=1,
                                               num_constraints=0,
                                               batch_size=None,
                                               batch_strategy='median_imputation',
                                               initial_trials=self.bo_init_num,
                                               init_strategy='random_explore_first',
                                               optimization_strategy='bo',
                                               surrogate_type='prf',
                                               acq_type='ei',
                                               acq_optimizer_type='local_random',
                                               task_id=self.method_name,
                                               output_dir=self.log_directory,
                                               random_state=random_state,
                                               logger_kwargs=_logger_kwargs,
                                               )
        self.config_advisor.optimizer.rand_prob = 0.0

    def choose_next(self, num_config):
        # Sample n configurations according to BOHB strategy.
        logger.info('Sample %d configs in choose_next. rand_prob is %f.' % (num_config, self.rand_prob))

        # get bo configs
        # update batchsize each round. random ratio is fixed.
        self.config_advisor.batch_size = num_config - int(num_config * self.rand_prob)
        bo_configs = self.config_advisor.get_suggestions()
        bo_configs = bo_configs[:num_config]  # may exceed num_config in initial random sampling
        logger.info('len bo configs = %d.' % len(bo_configs))

        # sample random configs
        configs = expand_configurations(bo_configs, self.config_space, num_config)
        logger.info('len total configs = %d.' % len(configs))
        assert len(configs) == num_config
        return configs

    def update_incumbent_before_reduce(self, T, val_losses, n_iteration):
        if int(n_iteration) < self.R:
            return
        self.incumbent_configs.extend(T)
        self.incumbent_perfs.extend(val_losses)
        # update config advisor
        for config, perf in zip(T, val_losses):
            objectives = [perf]
            observation = Observation(
                config=config, objectives=objectives, constraints=None,
                trial_state=SUCCESS, elapsed_time=None,
            )
            self.config_advisor.update_observation(observation)
            logger.info('update observation: config=%s, perf=%f' % (str(config), perf))
        logger.info('%d observations updated. %d incumbent configs total.' % (len(T), len(self.incumbent_configs)))

    def update_incumbent_after_reduce(self, T, incumbent_loss):
        return
