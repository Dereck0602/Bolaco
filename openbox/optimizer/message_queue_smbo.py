# License: MIT

import time
from typing import List
import numpy as np

from openbox import logger
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.optimizer.base import BOBase
from openbox.core.message_queue.master_messager import MasterMessager
from openbox.utils.history import History
from openbox.utils.util_funcs import deprecate_kwarg


class mqSMBO(BOBase):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    @deprecate_kwarg('time_limit_per_trial', 'max_runtime_per_trial', 'a future version')
    def __init__(
            self,
            objective_function,
            config_space,
            parallel_strategy='async',
            batch_size=4,
            batch_strategy='default',
            num_objectives=1,
            num_constraints=0,
            sample_strategy: str = 'bo',
            max_runs=100,
            max_runtime_per_trial=None,
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            initial_runs=3,
            init_strategy='random_explore_first',
            initial_configurations=None,
            ref_point=None,
            transfer_learning_history: List[History] = None,
            logging_dir='logs',
            task_id='OpenBox',
            random_state=None,
            advisor_kwargs: dict = None,
            logger_kwargs: dict = None,
            ip="",
            port=13579,
            authkey=b'abc',
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [np.inf] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, max_runtime_per_trial=max_runtime_per_trial,
                         transfer_learning_history=transfer_learning_history, logger_kwargs=logger_kwargs)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size
        max_queue_len = max(100, 3 * batch_size)
        self.master_messager = MasterMessager(ip, port, authkey, max_queue_len, max_queue_len)

        advisor_kwargs = advisor_kwargs or {}
        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space,
                                                   num_objectives=num_objectives,
                                                   num_constraints=num_constraints,
                                                   batch_size=batch_size,
                                                   batch_strategy=batch_strategy,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   init_strategy=init_strategy,
                                                   transfer_learning_history=transfer_learning_history,
                                                   optimization_strategy=sample_strategy,
                                                   surrogate_type=surrogate_type,
                                                   acq_type=acq_type,
                                                   acq_optimizer_type=acq_optimizer_type,
                                                   ref_point=ref_point,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   random_state=random_state,
                                                   logger_kwargs=_logger_kwargs,
                                                   **advisor_kwargs)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space,
                                                    num_objectives=num_objectives,
                                                    num_constraints=num_constraints,
                                                    batch_size=batch_size,
                                                    batch_strategy=batch_strategy,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    init_strategy=init_strategy,
                                                    transfer_learning_history=transfer_learning_history,
                                                    optimization_strategy=sample_strategy,
                                                    surrogate_type=surrogate_type,
                                                    acq_type=acq_type,
                                                    acq_optimizer_type=acq_optimizer_type,
                                                    ref_point=ref_point,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    random_state=random_state,
                                                    logger_kwargs=_logger_kwargs,
                                                    **advisor_kwargs)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

    def async_run(self):
        config_num = 0
        result_num = 0
        while result_num < self.max_runs:
            # Add jobs to masterQueue.
            while len(self.config_advisor.running_configs) < self.batch_size and config_num < self.max_runs:
                config_num += 1
                config = self.config_advisor.get_suggestion()
                msg = [config, self.max_runtime_per_trial, self.FAILED_PERF]
                logger.info("Master: Add config %d." % config_num)
                self.master_messager.send_message(msg)

            # Get results from workerQueue.
            while True:
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    # logger.info("Master: wait for worker results. sleep 1s.")
                    time.sleep(1)
                    break
                # Report result.
                result_num += 1
                self.config_advisor.update_observation(observation)
                logger.info('Master: Get %d observation: %s' % (result_num, str(observation)))

    def sync_run(self):
        batch_id, config_count = 0, 0
        while True:
            batch_id += 1
            configs = self.config_advisor.get_suggestions()
            # Add batch configs to masterQueue.
            for config in configs:
                msg = [config, self.max_runtime_per_trial, self.FAILED_PERF]
                self.master_messager.send_message(msg)
            logger.info('Master: %d-th batch. %d configs sent.' % (batch_id, len(configs)))
            # Get batch results from workerQueue.
            result_num = 0
            result_needed = len(configs)
            while True:
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    # logger.info("Master: wait for worker results. sleep 1s.")
                    time.sleep(1)
                    continue
                # Report result.
                result_num += 1
                if observation.objectives is None:
                    observation.objectives = self.FAILED_PERF.copy()
                self.config_advisor.update_observation(observation)
                logger.info('Master: In the %d-th batch [%d], observation is: %s'
                                 % (batch_id, result_num, str(observation)))
                if result_num == result_needed:
                    break
            config_count += result_needed
            if config_count >= self.max_runs:
                break

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
        return self.get_history()
