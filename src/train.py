
from datetime import datetime

import numpy as np

from config.config import Config
from src.models.algorithms.soft_actor_critic import SoftActorCritic
from src.utils.spherical_to_cartesian_coordinates import spherical_to_cartesian_coordinates
from src.utils.vector_functions import angle_between
from src.utils.satellite_functions import get_aods, get_steering_vec
from src.utils.real_complex_vector_reshaping import complex_vector_to_double_real_vector, real_vector_to_half_complex_vector, complex_vector_to_rad_and_phase
from src.utils.progress_printer import progress_printer


class Trainer:

    def __init__(self):

        config = Config()
        config.config_learner.algorithm_args['network_args']['num_actions'] = 2 * config.sat_ant_nr

        sac = SoftActorCritic(
            rng=config.rng,
            **config.config_learner.algorithm_args,
        )

        self.own_spherical_coordinates = np.array([10, np.pi / 2, np.pi / 2])
        self.users_spherical_coordinates = [
            np.array([1, np.pi / 2, np.pi / 2 + 0.2]),
            # np.array([1, np.pi/2, np.pi/2]),
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
        value_loss = None
        mean_log_prob_density = None

        real_time_start = datetime.now()

        state = complex_vector_to_rad_and_phase(np.array(self.steering_vecs).flatten())
        step_experience['state'] = state

        best_reward = 0
        best_precoder = None
        best_action = None
        best_reward_training_step = None

        for training_step in range(training_steps):

            action = sac.get_action(state)
            step_experience['action'] = action
            # print(action)
            # print(action % (2*np.pi))
            # print(state)
            # exit()

            # action = np.array([1.42*np.pi, 0.6*np.pi, 0, 0])
            w_precoder_vector = np.exp(1j * (action % (2*np.pi)))
            # w_precoder_vector = np.exp(1j * np.clip(action, a_min=-np.pi, a_max=np.pi))
            # w_precoder_vector = np.exp(1j * action * np.pi)
            w_precoder = w_precoder_vector.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))
            # w_precoder = np.array(
            #     [[-0.55557023 - 0.83146961j, -0.46174861 + 0.88701083j],
            #      [1. + 0.j          ,1. + 0.j]]
            # )
            # print(w_precoder)
            # exit()

            # print('\n', w_precoder)

            norm_factor = np.sqrt(1 / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
            normalized_precoder = norm_factor * w_precoder

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
            # print(power_gains_users)
            # print(signal_to_interference_ratio_per_user)
            # exit()

            reward = sum(signal_to_interference_ratio_per_user)
            step_experience['reward'] = reward
            training_reward[training_step] = reward

            if reward > best_reward:
                best_reward = reward
                best_reward_training_step = training_step
                best_precoder = normalized_precoder.copy()
                best_action = action % (2*np.pi)
                # best_action = np.clip(action, a_min=-np.pi, a_max=np.pi)
            # print('\n', reward)

            sac.add_experience(experience=step_experience)

            train_policy = config.config_learner.policy_training_criterion(simulation_step=training_step)
            train_value = config.config_learner.value_training_criterion(simulation_step=training_step)

            if train_value or train_policy:
                mean_log_prob_density, value_loss = sac.train(
                    toggle_train_value_networks=True,
                    toggle_train_policy_network=True,
                    toggle_train_entropy_scale_alpha=True,
                )

            if training_step % 100 == 0:
                print('\n', np.nanmean(training_reward[max(0, training_step-100):training_step]))
                if value_loss:
                    pass
                    # print('\n', value_loss)
                    # print('\n', mean_log_prob_density)

            progress_printer(
                progress=(training_step+1) / training_steps,
                real_time_start=real_time_start,
            )

        # print('\n', training_reward)
        print('\n', best_reward)
        print('\n', best_precoder)
        print('\n', best_action)
        print('\n', best_reward_training_step)


if __name__ == '__main__':
    trainer = Trainer()
