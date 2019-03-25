import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

from baselines import bench

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
from baselines.atari_wrapper import AtariWrapper, PacmanWrapper
import os


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)

class QNetworkTrainingWrapper(object):
    def __init__(self, env,
              network,
              scope='deepq',
              seed=None,
              lr=5e-4,
              total_timesteps=100000,
              buffer_size=50000,
              exploration_fraction=0.1,
              exploration_final_eps=0.02,
              train_freq=1,
              batch_size=32,
              print_freq=100,
              gpu_num=-1,
              checkpoint_freq=10000,
              checkpoint_path=None,
              learning_starts=1000,
              gamma=1.0,
              target_network_update_freq=500,
              prioritized_replay=False,
              prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4,
              prioritized_replay_beta_iters=None,
              prioritized_replay_eps=1e-6,
              param_noise=False,
              callback=None,
              multihead=False,
              num_heads=1,
              load_path=None,
              **network_kwargs
              ):
        """Train a deepq model.

        Parameters
        -------
        env: gym.Env
            environment to train on
        q_func: (tf.Variable, int, str, bool) -> tf.Variable
            the model that takes the following inputs:
                observation_in: object
                    the output of observation placeholder
                num_actions: int
                    number of actions
                scope: str
                reuse: bool
                    should be passed to outer variable scope
            and returns a tensor of shape (batch_size, num_actions) with values of every action.
        lr: float
            learning rate for adam optimizer
        total_timesteps: int
            number of env steps to optimizer for
        buffer_size: int
            size of the replay buffer
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
            set to None to disable printing
        batch_size: int
            size of a batched sampled from replay buffer for training
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        checkpoint_freq: int
            how often to save the model. This is so that the best version is restored
            at the end of the training. If you do not wish to restore the best version at
            the end of the training set this variable to None.
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to total_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.
        callback: (locals, globals) -> None
            function called at every steps with state of the algorithm.
            If callback returns true training stops.
        load_path: str
            path to load the model from. (default: None)
        **network_kwargs
            additional keyword arguments to pass to the network builder.

        Returns
        -------
        act: ActWrapper
            Wrapper over act function. Adds ability to save it and load it.
            See header of baselines/deepq/categorical.py for details on the act function.
        """
        # Create all the functions necessary to train the model
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.param_noise = param_noise
        self.print_freq = print_freq
        self.total_timesteps = total_timesteps
        self.callback = callback
        self.env = env
        use_gpu = gpu_num != -1
        config = None
        #if use_gpu:
        #    config = tf.ConfigProto(device_count = {'GPU': gpu_num})
        sess = get_session(num_gpu=gpu_num)
        set_global_seeds(seed)

        q_func = q_func = build_q_func(network, multihead=multihead, num_heads=num_heads, **network_kwargs)

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph

        observation_space = env.observation_space

        def make_obs_ph(name):
            return ObservationInput(observation_space, name=name)


        with tf.device(f'/{"gpu" if use_gpu else "cpu"}:{gpu_num if use_gpu else 0}'):
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=make_obs_ph,
                q_func=q_func,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                gamma=gamma,
                grad_norm_clipping=10,
                param_noise=param_noise,
                scope=scope,
                multihead=multihead,
                num_heads=num_heads
            )
        self.train = train
        self.update_target = update_target
        self.q_values = debug['q_values']
        self.saver = debug['saver']


        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
        }

        self.act = act = ActWrapper(act, act_params)

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None

        self.beta_schedule = beta_schedule
        self.replay_buffer = replay_buffer

        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)
        self.exploration = exploration

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        self.episode_rewards = episode_rewards = [0.0]
        saved_mean_reward = None

        def get_batch_func(time):
            if self.prioritized_replay:
                experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(time))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            return obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes

        self.get_batch_func = get_batch_func


    def train_network(self):

        obs = env.reset()
        reset = True

        for t in range(self.total_timesteps):
            if self.callback is not None:
                if self.callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            # compute the action of the agent while annealing associated epsilon values.
            action = self.get_action_with_side_effects(t, reset, obs)

            env_action = action
            reset = False
            new_obs, rew, done, info = env.step(env_action)
            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, rew, new_obs, float(done))
            if 'internal_terminal' in info:
                internal_done = info['internal_terminal']
            else:
                internal_done = done
            obs = new_obs

            self.episode_rewards[-1] += rew
            if done:
                obs = env.reset()
            if internal_done:
                print(len(self.episode_rewards))
                self.episode_rewards.append(0.0)
                reset = True

            # runs an iteration of the training procedure.
            self.run_step(t, self.get_batch_func)
            self.perform_logging(t, internal_done)


    def train_batch(self, time, obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes, reward_nums):
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        td_errors = self.train(obses_t, actions, rewards, obses_tp1, dones, weights, reward_nums)
        loss = np.mean(np.square(td_errors))
        #if self.prioritized_replay:
        #    new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
        #    self.replay_buffer.update_priorities(batch_idxes, new_priorities)
        #else:
        #    td_errors = None
        if time % self.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()
        return loss


    def get_action_with_side_effects(self, time, reset, single_state):
        kwargs = {}
        if not self.param_noise:
            update_eps = self.exploration.value(time)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(
                1. - self.exploration.value(time) + self.exploration.value(time) / float(self.env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        action = self.act(np.array(single_state)[None], update_eps=update_eps, **kwargs)[0]
        return action

    def perform_logging(self, time, internal_done):
        mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 1)
        num_episodes = len(self.episode_rewards)
        if internal_done and self.print_freq is not None and len(self.episode_rewards) % self.print_freq == 0:
            print(
                f'steps {time}\nepisodes {num_episodes}\nmean 100 episode reward {mean_100ep_reward}\n% time spent exploring {int(100 * self.exploration.value(time))}')
            logger.record_tabular("steps", time)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(time)))
            logger.dump_tabular()


    def get_action(self, state, reward_num):
        return self.act(state, reward_num, stochastic=False, update_eps=-1)

    def get_Q(self, state, reward_num):
        return self.q_values(state, reward_num)

    def save(self, path, name):
        self.saver.save(get_session(), os.path.join(path, name))

    def restore(self, path, name):
        self.saver.restore(get_session(), os.path.join(path, name))

    def clean(self):
        sess = get_session()
        sess.__enter__()
        tf.reset_default_graph()
        sess.__exit__(None, None, None)


def make_dqn(env, scope, gpu_num, multihead=False, num_heads=1):
    return QNetworkTrainingWrapper(
        env,
        "conv_only",
        scope=scope,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        gpu_num=gpu_num,
        total_timesteps=int(1e7),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        multihead=multihead,
        num_heads=num_heads,
        gamma=0.99)




if __name__ == '__main__':
    env = PacmanWrapper()
    env = bench.Monitor(env, logger.get_dir())
    QNetworkTrainingWrapper(env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99).train_network()