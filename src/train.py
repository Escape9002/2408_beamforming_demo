
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from config.config import Config
from src.models.algorithms.soft_actor_critic import SoftActorCritic
from src.utils.spherical_to_cartesian_coordinates import spherical_to_cartesian_coordinates
from src.utils.vector_functions import angle_between
from src.utils.satellite_functions import get_aods, get_steering_vec
from src.utils.real_complex_vector_reshaping import complex_vector_to_double_real_vector, real_vector_to_half_complex_vector, complex_vector_to_rad_and_phase
from src.utils.progress_printer import progress_printer
from src.utils.profiling import start_profiling, end_profiling


def transform_action(action):
    # w_precoder_vector = np.exp(1j * (action % (2*np.pi)))
    # w_precoder_vector = np.exp(1j * np.clip(action, a_min=-np.pi, a_max=np.pi))
    # w_precoder_vector = np.exp(1j * action * np.pi)
    # w_precoder_vector = np.tanh(action) * np.pi
    # w_precoder_vector = (0.5 * np.tanh(action) + 0.5) * 2 * np.pi
    w_precoder_vector = action * np.pi

    w_precoder_vector = np.exp(1j * w_precoder_vector)

    return w_precoder_vector


class Trainer:

    def __init__(self):

        config = Config()
        config.config_learner.algorithm_args['network_args']['num_actions'] = config.sat_ant_nr * config.user_nr

        sac = SoftActorCritic(
            rng=config.rng,
            **config.config_learner.algorithm_args,
        )

        self.own_spherical_coordinates = np.array([10, np.pi / 2, np.pi / 2])
        self.users_spherical_coordinates = [
            np.array([1, np.pi / 2, np.pi / 2 + 0.2]),
            np.array([1, np.pi/2, np.pi/2]),
            np.array([1, np.pi / 2, np.pi / 2 - 0.2]),
        ]

        training_steps = 100_000

        self.user_aods = get_aods(
            own_spherical_coordinates=self.own_spherical_coordinates,
            users_spherical_coordinates=self.users_spherical_coordinates,
        )

        self.steering_vecs = [
            get_steering_vec(
                antenna_num=config.sat_ant_nr,
                antenna_distance=config.sat_ant_dist,
                wavelength=config.wavelength,
                cos_aod=np.cos(user_aod),
            )
            for user_aod in self.user_aods
        ]

        training_reward = np.zeros(training_steps)
        step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
        value_losses = []
        mean_log_prob_densities = []
        direct_inference_rewards = []
        steps_with_training = []

        real_time_start = datetime.now()

        state = complex_vector_to_rad_and_phase(np.array(self.steering_vecs).flatten())
        # state = np.random.normal(loc=0, scale=1, size=state.shape)
        step_experience['state'] = state

        best_reward = -np.infty
        best_action = None
        best_reward_training_step = None
        random_or_nah = None

        profiler = start_profiling()

        for training_step in range(training_steps):

            # state = np.random.normal(loc=0, scale=1, size=state.shape)
            # state = np.zeros(shape=state.shape)

            action = sac.get_action(state)
            step_experience['action'] = action

            w_precoder_vector = transform_action(action)
            w_precoder = w_precoder_vector.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))

            norm_factor = np.sqrt(1 / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
            normalized_precoder = norm_factor * w_precoder

            reward = self.test_precoder(normalized_precoder)

            step_experience['reward'] = reward
            training_reward[training_step] = reward

            if reward > best_reward:
                best_reward = reward
                best_reward_training_step = training_step
                best_action = action
                random_or_nah = 'random'
                print(f'new best reward {reward}, {random_or_nah}')

            sac.add_experience(experience=step_experience)

            train_policy = config.config_learner.policy_training_criterion(simulation_step=training_step)
            train_value = config.config_learner.value_training_criterion(simulation_step=training_step)

            if train_value or train_policy:
                mean_log_prob_density, value_loss = sac.train(
                    toggle_train_value_networks=True,
                    toggle_train_policy_network=True,
                    toggle_train_entropy_scale_alpha=True,
                )
                mean_log_prob_densities.append(mean_log_prob_density)
                value_losses.append(value_loss)

                # evaluate raw precoder
                action, _ = sac.networks['policy'][0]['primary'].call(state[np.newaxis])
                action = action.numpy().flatten()
                w_precoder_vector = transform_action(action)
                w_precoder = w_precoder_vector.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))
                norm_factor = np.sqrt(1 / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
                normalized_precoder = norm_factor * w_precoder
                reward = self.test_precoder(normalized_precoder)
                # print(' ', reward)
                direct_inference_rewards.append(reward)
                steps_with_training.append(training_step)
                if reward > best_reward:
                    best_reward = reward
                    best_reward_training_step = training_step
                    best_action = action
                    random_or_nah = 'nah'
                    print(f'new best reward {reward}, {random_or_nah}')

            if (training_step+1) % int(training_steps/20) == 0:
                print('\n', np.nanmean(training_reward[max(0, training_step-int(training_steps/20)):training_step]))

            progress_printer(
                progress=(training_step+1) / training_steps,
                real_time_start=real_time_start,
            )

        # print('\n', training_reward)
        print('\n', best_reward, random_or_nah)
        print('\n', best_action)
        print('\n', best_reward_training_step)

        plt.plot(steps_with_training, mean_log_prob_densities, label='mean logpis')
        plt.plot(steps_with_training, value_losses, label='value loss')
        plt.plot(steps_with_training, direct_inference_rewards, label='direct_inference_rewards')
        smoothed_training_rewards = [
            np.mean(training_reward[max(0, training_step-int(training_steps/20)):training_step])
            for training_step in range(training_steps)
        ]
        plt.plot(range(training_steps), smoothed_training_rewards, label='training rewards')
        plt.legend()
        plt.show()

        # end_profiling(profiler)

    def test_precoder(self, normalized_precoder):

        power_gains_users = np.zeros((len(self.steering_vecs), len(self.users_spherical_coordinates)))
        for steering_vec_id, steering_vec in enumerate(self.steering_vecs):
            for user_id in range(len(self.users_spherical_coordinates)):
                power_gain_user = abs(np.matmul(steering_vec, normalized_precoder[:, user_id])) ** 2
                power_gains_users[steering_vec_id, user_id] = power_gain_user

        signal_to_interference_ratio_per_user = np.zeros(len(self.users_spherical_coordinates))
        for user_id in range(len(self.users_spherical_coordinates)):
            signal_to_interference_ratio_per_user[user_id] = (
                power_gains_users[user_id, user_id] / (
                    np.sum(np.delete(power_gains_users[user_id, :], user_id, axis=0), axis=0)
                    + 0.01  # regularizing noise
                )
            )

        signal_to_interference_ratio_per_user = np.log10(signal_to_interference_ratio_per_user)

        reward = sum(signal_to_interference_ratio_per_user)
        reward_nonzero = sum(np.maximum(0, signal_to_interference_ratio_per_user))

        # reward_nonzero = -np.infty
        # for user_id in range(len(self.user_aods)):    # scuffed way to avoid sparse rewards
        #     if signal_to_interference_ratio_per_user[user_id] > reward_nonzero:
        #         reward_nonzero = signal_to_interference_ratio_per_user[user_id]

        return reward
        # return max(reward, reward_nonzero)
        # return reward_nonzero


if __name__ == '__main__':
    trainer = Trainer()
