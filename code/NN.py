"""
@author: Benjamin Virrion
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
#import copy
import math
import time
import pandas as pd
from tempfile import TemporaryFile
import random
import tensorflow as tf
import json
# from numba import jit
# import cProfile
from tensorflow.python.client import timeline

class NeuralNetwork():
    def __init__(self, params):
        print("Starting NeuralNetwork.__init__()")
        start_init = time.time()
        self.params = params
        self.test_1 = tf.constant(1)
        self.test_2 = tf.constant(1)
        self.test_3 = tf.constant(1)
        self.test_a_1 = tf.constant(1)
        self.N_iter_show_proportion = params["NN"]["NIterShowProportion"]
        self.N_batch_change_proportion = params["NN"]["NBatchChangeProportion"]
        self.N_batch_size = params["NN"]["NBatchSize"]
        self.N_time = params["PayoffChars"]["NTime"]
        if self.params["Payoff"] == "Call" or self.params["Payoff"] == "Calls&Puts":
            self.K = params["PayoffChars"]["K"]
        if self.params["Payoff"] == "Calls&Puts":
            self.K_2 = params["PayoffChars"]["K_2"]
            self.number_of_calls = params["PayoffChars"]["NumberOfCalls"]
            self.number_of_puts = params["PayoffChars"]["NumberOfPuts"]
        # self.sigma = params["Diffusion"]["Sigma"]
        self.Sigma = tf.placeholder(tf.float64, shape=[])
        self.T = params["PayoffChars"]["T"]
        self.dt = self.T / self.N_time
        self.sqrt_dt = np.sqrt(self.dt)
        # self.x = params["Diffusion"]["X"]
        self.x = tf.placeholder(tf.float64, shape=[])
        self.A = tf.placeholder(tf.float64, shape=[])
        self.B = tf.placeholder(tf.float64, shape=[])
        self.M = tf.placeholder(tf.float64, shape=[])
        self.Rho = tf.placeholder(tf.float64, shape=[])
        self.lambda_constraint = params["NN"]["LambdaConstraint"]
        self.constraint = params["NN"]["Constraint"]

        """
        self.trajectories = tf.placeholder(tf.float64, shape=[self.N_batch_size, self.N_time + 1])
        """

        self.random_gaussians = tf.placeholder(tf.float64, shape=[self.N_batch_size, self.N_time])
        self.layers_list = self.generate_layers_list()
        self.weights_list, self.biases_list = self.generate_weights_list_and_biases_list()

        self.trajectories, self.z, self.neural_nets_list = self.generate_trajectories_z_and_neural_nets_list()

        # self.test_2 = self.trajectories[self.N_time - 1]

        # self.neural_nets_list = self.generate_neural_nets_list()

        self.payoff_values = self.generate_payoff_values()
        # self.test_2 = self.payoff_values
        # self.test_1 = self.payoff_values

        self.price = self.generate_price()

        # self.test_1 = self.trajectories
        # self.test_2 = self.random_gaussians
        # self.test_3 = tf.multiply(self.payoff_values, self.z)

        # self.KL_divergence = self.generate_KL_divergence()
        # self.test_1 = self.KL_divergence

        self.variance = self.generate_variance()

        self.variance_z = self.generate_variance_z()

        self.test_1 = self.variance_z

        self.std = tf.sqrt(self.variance)

        self.loss = self.generate_loss()

        self.sess = tf.Session()
        self.learning_rate = tf.placeholder(tf.float64, shape=[])

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # initialize session and variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        end_init = time.time()
        print("Time spent in NeuralNetwork.__init__(): ", end_init - start_init)
        print("End of NeuralNetwork.__init__()")

    def generate_KL_divergence(self):
        return_value = tf.reduce_sum(tf.log(self.z)) / self.N_batch_size
        return return_value

    def monte_carlo_price(self, number_of_batches):
        random_gaussians = np.random.normal(size=(self.N_batch_size * number_of_batches, self.N_time))
        trajectories = np.full((self.N_batch_size * number_of_batches, self.N_time + 1), 0.)
        if self.params["Diffusion"]["Type"] == "LV":
            x = np.log(self.params["Diffusion"]["X"])
        else:
            x = self.params["Diffusion"]["X"]
        trajectories[:, 0] = np.full((self.N_batch_size * number_of_batches), x)
        for t_step in range(self.N_time):
            if self.params["Diffusion"]["Type"] == "LV":
                trajectories[:, t_step + 1] = trajectories[:, t_step] + self.sigma_mc(trajectories[:, t_step], t_step + 1) * random_gaussians[:, t_step] * self.sqrt_dt - 0.5 * np.square(self.sigma_mc(trajectories[:, t_step], t_step + 1)) * self.dt
            else:
                trajectories[:, t_step + 1] = trajectories[:, t_step] + self.sigma_mc(trajectories[:, t_step], t_step + 1) * random_gaussians[:, t_step] * self.sqrt_dt
        if self.params["Diffusion"]["Type"] == "LV":
            trajectories = np.exp(trajectories)
        payoff_values = self.monte_carlo_payoff(trajectories)
        price = np.average(payoff_values)
        std = np.std(payoff_values) / np.sqrt(self.N_batch_size * number_of_batches)

        return price, std

    def sigma_mc(self, k_batch, t_step):
        if self.params["Diffusion"]["Type"] == "LV":
            if t_step == 0:
                return np.full((np.shape(k_batch)[0]), 0., dtype=np.float64)
            else:
                a = np.float64(self.params["Diffusion"]["A"])
                b = np.float64(self.params["Diffusion"]["B"])
                rho = np.float64(self.params["Diffusion"]["Rho"])
                m = np.float64(self.params["Diffusion"]["M"])
                sigma = np.float64(self.params["Diffusion"]["Sigma"])
                term_in_sqrt_term = np.square(k_batch - m) + np.square(sigma)
                sqrt_term = np.sqrt(term_in_sqrt_term)

                partial_t_w = a + b * (rho * (k_batch - m) + sqrt_term)
                w = t_step * self.dt * partial_t_w
                partial_k_w = t_step * self.dt * b * (np.divide(k_batch - m, sqrt_term))
                partial_kk_w = np.divide(t_step * self.dt * b * np.square(sigma), np.power(term_in_sqrt_term, 1./3.))

                denominator_first_term = 1 - k_batch / w * partial_k_w
                denominator_second_term = 0.25 * (- 0.25 - 1./w + np.divide(np.square(k_batch), np.square(w))) * np.square(partial_k_w)
                denominator_third_term = 0.5 * partial_kk_w

                sigma_square = np.divide(partial_t_w, denominator_first_term + denominator_second_term + denominator_third_term)

                sigma = np.sqrt(sigma_square)

                """
                if t_step == 1:
                    self.test_2 = [sigma, sigma_square, term_in_sqrt_term, k_batch, partial_t_w, w, partial_k_w, partial_kk_w, denominator_first_term, denominator_second_term, denominator_third_term]
                """

        else:
            sigma = self.params["Diffusion"]["Sigma"]

        return sigma


    def sigma(self, k_batch, t_step):
        if self.params["Diffusion"]["Type"] == "LV":
            # a = np.float64(self.params["Diffusion"]["A"])
            a = self.A
            # b = np.float64(self.params["Diffusion"]["B"])
            b = self.B
            # rho = np.float64(self.params["Diffusion"]["Rho"])
            rho = self.Rho
            # m = np.float64(self.params["Diffusion"]["M"])
            m = self.M
            # sigma = np.float64(self.params["Diffusion"]["Sigma"])
            sigma = self.Sigma
            term_in_sqrt_term = tf.square(k_batch - m) + tf.square(sigma)
            sqrt_term = tf.math.sqrt(term_in_sqrt_term)
            partial_t_w = a + b * (rho * (k_batch - m) + sqrt_term)

            if t_step == 0:
                partial_k_over_omega_numerator = b * (rho + (k_batch - m)/sqrt_term)
                partial_k_over_omega_denominator = a + b * (rho * (k_batch - m) + sqrt_term)
                partial_k_over_omega = partial_k_over_omega_numerator / partial_k_over_omega_denominator

                denominator_first_term = 1 - k_batch * partial_k_over_omega
                denominator_second_term = 0.25 * (-0.25 + tf.square(k_batch * partial_k_over_omega))

                sigma_square = tf.divide(partial_t_w, denominator_first_term + denominator_second_term)
                sigma = tf.sqrt(sigma_square)
                # return np.full((np.shape(k_batch)[0], 1), 0., dtype=np.float64)
                # self.test_3 = sigma
                return sigma
            else:
                w = t_step * self.dt * partial_t_w
                partial_k_w = t_step * self.dt * b * (tf.divide(k_batch - m, sqrt_term))
                partial_kk_w = tf.divide(t_step * self.dt * b * tf.square(sigma), tf.pow(term_in_sqrt_term, 1./3.))

                denominator_first_term = 1 - k_batch / w * partial_k_w
                denominator_second_term = 0.25 * (- 0.25 - 1./w + tf.divide(tf.square(k_batch), tf.square(w))) * tf.square(partial_k_w)
                denominator_third_term = 0.5 * partial_kk_w

                sigma_square = tf.divide(partial_t_w, denominator_first_term + denominator_second_term + denominator_third_term)

                sigma = tf.sqrt(sigma_square)

                """
                if t_step == 1:
                    self.test_2 = [sigma, sigma_square, term_in_sqrt_term, k_batch, partial_t_w, w, partial_k_w, partial_kk_w, denominator_first_term, denominator_second_term, denominator_third_term]
                """

        else:
            # sigma = self.params["Diffusion"]["Sigma"]
            sigma = self.Sigma

        return sigma


    def monte_carlo_payoff(self, trajectories):
        if self.params["Payoff"] == "AutoCall":
            number_phoenix_dates = len(self.params["PayoffChars"]["PhoenixDates"])
            phoenix_autocall_spreads = np.full((trajectories.shape[0], number_phoenix_dates), 0.)
            phoenix_autocall_spreads_proba = np.full((trajectories.shape[0], number_phoenix_dates), 0.)
            phoenix_autocall_stay_spreads_previous_proba_list = np.full((trajectories.shape[0], number_phoenix_dates), 1.)
            phoenix_dates = self.params["PayoffChars"]["PhoenixDates"]
            phoenix_autocall_spreads_chars = self.params["PayoffChars"]["PhoenixAutoCallSpreads"]
            phoenix_coupons_chars = self.params["PayoffChars"]["PhoenixCoupons"]
            pdi_chars = self.params["PayoffChars"]["PDI"]

            for i in range(number_phoenix_dates):
                cur_autocall_spread = np.maximum(trajectories[:, phoenix_dates[i]]
                                                 - phoenix_autocall_spreads_chars[i][0], 0) \
                                      - np.maximum(trajectories[:, phoenix_dates[i]]
                                                   - phoenix_autocall_spreads_chars[i][1], 0)
                cur_autocall_spread *= 1./(phoenix_autocall_spreads_chars[i][1] - phoenix_autocall_spreads_chars[i][0])
                phoenix_autocall_spreads_proba[:, i] = cur_autocall_spread

                if i > 0:
                    phoenix_autocall_stay_spreads_previous_proba_list[:, i] = \
                        phoenix_autocall_stay_spreads_previous_proba_list[:, i - 1] * (1. - cur_autocall_spread)
            phoenix_autocall_spreads = np.multiply(
                phoenix_autocall_spreads_proba,
                phoenix_autocall_stay_spreads_previous_proba_list)

            phoenix_coupons_2D = np.repeat(phoenix_coupons_chars, trajectories.shape[0], axis=0)
            phoenix_coupons = np.expand_dims(np.sum(np.multiply(phoenix_autocall_spreads, phoenix_coupons_2D), axis=-1), axis=-1)


            # phoenix_autocall_spreads = tf.reduce_sum(tf.stack(phoenix_autocall_spreads_list, axis=1), axis=2)
            # phoenix_coupons_2D = np.repeat(phoenix_coupons_chars, trajectories.shape[0], axis=0)
            # phoenix_coupons = np.multiply(phoenix_autocall_spreads, phoenix_coupons_2D)

            number_of_puts_in_spread = (1. - pdi_chars[0])/(pdi_chars[1] - pdi_chars[0])
            pdi = (number_of_puts_in_spread + 1.) * np.maximum(pdi_chars[1] - trajectories[:, self.N_time], 0) \
                  - number_of_puts_in_spread * np.maximum(pdi_chars[0] - trajectories[:, self.N_time], 0)
            pdi = np.expand_dims(pdi, axis=-1)

            proba_of_non_autocalled = np.expand_dims(np.multiply(
                phoenix_autocall_stay_spreads_previous_proba_list[:, number_phoenix_dates - 1],
                (1. - phoenix_autocall_spreads_proba[:, number_phoenix_dates - 1])), axis=-1)


            # proba_of_non_autocalled = np.expand_dims((1. - np.max(phoenix_autocall_spreads, axis=1)), axis=-1)
            return_value = phoenix_coupons - np.multiply(pdi, proba_of_non_autocalled)
        elif self.params["Payoff"] == "Calls&Puts":
            # empty_0 = np.full(trajectories.shape[0], 0.)
            return_value = self.number_of_calls * np.maximum(trajectories[:, self.N_time] - self.K, 0.) \
                           + self.number_of_puts * np.maximum(self.K_2 - trajectories[:, self.N_time], 0.)
        elif self.params["Payoff"] == "Call":
            return_value = np.maximum(trajectories[:, self.N_time] - self.K, 0.)
        return return_value

    def bachelier_call_price(self):
        if self.params["Diffusion"]["Type"] == "LV":
            return_value = "not available"
        else:
            sigma = self.params["Diffusion"]["Sigma"]
            if self.params["Payoff"] == "AutoCall":
                return_value = "not available"
            elif self.params["Payoff"] == "Calls&Puts":
                """
                d_1 = (np.log(1./self.K) + np.square(self.sigma) / 2. * self.T) / (self.sigma * self.T)
                d_2 = d_1 - self.sigma * np.sqrt(self.T)
                call_price = 1. * scipy.stats.norm.cdf(d_1) - self.K * scipy.stats.norm.cdf(d_2)
                """
                x = self.params["Diffusion"]["X"]
                sigma_sqrt_T = sigma * np.sqrt(self.T)
                call_price = (x - self.K) * scipy.stats.norm.cdf((x - self.K) / sigma_sqrt_T) \
                             + sigma_sqrt_T * scipy.stats.norm.pdf((x - self.K) / sigma_sqrt_T)
                put_price = (self.K_2 - x) * scipy.stats.norm.cdf((self.K_2 - x) / sigma_sqrt_T) \
                             + sigma_sqrt_T * scipy.stats.norm.pdf((self.K_2 - x) / sigma_sqrt_T)
                return_value = self.number_of_calls * call_price + self.number_of_puts * put_price
            elif self.params["Payoff"] == "Call":
                x = self.params["Diffusion"]["X"]
                sigma_sqrt_T = sigma * np.sqrt(self.T)
                call_price = (x - self.K) * scipy.stats.norm.cdf((x - self.K) / sigma_sqrt_T) \
                             + sigma_sqrt_T * scipy.stats.norm.pdf((x - self.K) / sigma_sqrt_T)
                return_value = call_price
        return return_value

    def eval_a_on_grid(self, x_min, x_steps, x_step_size):
        if self.params["Diffusion"]["Type"] == "Bachelier":
            tf_dict = {self.x: self.params["Diffusion"]["X"],
                       self.Sigma: self.params["Diffusion"]["Sigma"]}
        elif self.params["Diffusion"]["Type"] == "LV":
            tf_dict = {self.x: self.params["Diffusion"]["X"],
                       self.A: self.params["Diffusion"]["A"],
                       self.B: self.params["Diffusion"]["B"],
                       self.Rho: self.params["Diffusion"]["Rho"],
                       self.M: self.params["Diffusion"]["M"],
                       self.Sigma: self.params["Diffusion"]["Sigma"]}
        self.a_list_on_grid = self.construct_a_on_grid(x_min, x_steps, x_step_size)
        a_list, test_a_1 = self.sess.run([self.a_list_on_grid, self.test_a_1], tf_dict)
        print("test_a_1 = ", test_a_1)
        a_reduced_list = [None for i in range(self.N_time)]
        for n_time_step in range(self.N_time):
            a_reduced_list[n_time_step] = a_list[n_time_step][:x_steps]
        return a_reduced_list

    def construct_a_on_grid(self, x_min, x_steps, x_step_size):
        a_list = [None for i in range(self.N_time)]
        for t_step in range(self.N_time):
            trajectories = self.construct_linear_trajectories_on_x_grid_at_t_step(x_min, x_steps, x_step_size, t_step)
            a_list[t_step] = self.construct_a_on_trajectories_at_t_step(trajectories, t_step)
        return a_list

    def construct_linear_trajectories_on_x_grid_at_t_step(self, x_min, x_steps, x_step_size, t_step):
        trajectories = [None for i in range(t_step + 1)]
        for n_time_step in range(t_step + 1):
            if n_time_step == 0:
                # trajectories[0] = tf.constant(self.x, dtype=tf.float64, shape=(self.N_batch_size, 1))
                trajectories[0] = tf.tile(tf.reshape(self.x, shape=[1, 1]), [self.N_batch_size, 1])
            else:
                current_tensor_list = [None for i in range(x_steps + 1)]
                for x_step in range(x_steps):
                    # current_tensor_list[x_step] = tf.constant(self.x, dtype=tf.float64, shape=(1,)) + (x_min + x_step * x_step_size - self.x) * np.float(n_time_step) / np.float(t_step)
                    current_tensor_list[x_step] = tf.reshape(self.x, shape=[1]) + (x_min + x_step * x_step_size - self.x) * np.float(n_time_step) / np.float(t_step)
                # current_tensor_list[x_steps] = tf.constant(self.x, dtype=tf.float64, shape=(self.N_batch_size - x_steps,))
                current_tensor_list[x_steps] = tf.tile(tf.reshape(self.x, [1]), [self.N_batch_size - x_steps])
                trajectories[n_time_step] = tf.expand_dims(tf.concat(current_tensor_list, axis=0), -1)
        return trajectories

    def construct_a_on_trajectories_at_t_step(self, trajectories, t_step):
        if self.params["a"] == "full":
            a = self.neural_net(trajectories[:t_step + 1],
                                self.weights_list[t_step], # WAZA
                                self.biases_list[t_step],
                                t_step=t_step) # WAZA
        elif self.params["a"] == "local":
            a = self.neural_net([trajectories[t_step]],
                                self.weights_list[t_step],
                                self.biases_list[t_step],
                                t_step=t_step)
        return a


    def generate_trajectories_z_and_neural_nets_list(self):
        # sigma = self.params["Diffusion"]["Sigma"]
        a_list = [None for i in range(self.N_time)]
        trajectories = [None for i in range(self.N_time + 1)]
        neural_nets_list = [None for i in range(self.N_time)]
        for n_time_step in range(self.N_time + 1):
            if n_time_step == 0:
                if self.params["Diffusion"]["Type"] == "LV":
                    x = tf.log(self.x)
                else:
                    x = self.x
                # trajectories[n_time_step] = np.full((self.N_batch_size, 1), x, dtype=np.float64)
                # trajectories[n_time_step] = tf.constant(x, dtype=tf.float64, shape=(self.N_batch_size, 1))
                trajectories[n_time_step] = tf.tile(tf.reshape(x, [1, 1]), [self.N_batch_size, 1])
                """
                trajectories[n_time_step] = x + a_list[n_time_step] * self.sigma(trajectories[n_time_step-1], n_time_step) * self.dt \
                                            + gaussian_term * self.sigma(trajectories[n_time_step-1], n_time_step) * self.sqrt_dt
                """
            else:
                if self.params["a"] == "full":
                    neural_nets_list[n_time_step-1] = self.neural_net(trajectories[:n_time_step],
                                                                      self.weights_list[n_time_step - 1],
                                                                      self.biases_list[n_time_step - 1],
                                                                      t_step=n_time_step - 1)
                elif self.params["a"] == "local":
                    neural_nets_list[n_time_step-1] = self.neural_net([trajectories[n_time_step - 1]],
                                                                      self.weights_list[n_time_step - 1],
                                                                      self.biases_list[n_time_step - 1],
                                                                      t_step=n_time_step - 1)
                a_list[n_time_step - 1] = neural_nets_list[n_time_step - 1]
                gaussian_term = tf.expand_dims(self.random_gaussians[:, n_time_step - 1], axis=-1)
                if self.params["Diffusion"]["Type"] == "LV":
                    trajectories[n_time_step] = trajectories[n_time_step - 1] + a_list[n_time_step - 1] * self.sigma(trajectories[n_time_step-1], n_time_step - 1) * self.dt \
                                                + gaussian_term * self.sigma(trajectories[n_time_step - 1], n_time_step - 1) * self.sqrt_dt \
                                                - 0.5 * tf.square(self.sigma(trajectories[n_time_step - 1], n_time_step - 1)) * self.dt
                else:
                    trajectories[n_time_step] = trajectories[n_time_step - 1] + a_list[n_time_step - 1] * self.sigma(trajectories[n_time_step-1], n_time_step - 1) * self.dt \
                                                + gaussian_term * self.sigma(trajectories[n_time_step - 1], n_time_step - 1) * self.sqrt_dt

        a_squared_list = tf.square(tf.reduce_sum(tf.stack(a_list, axis=1), axis=2))
        # self.test_2 = a_list
        a_times_gaussians = tf.multiply(tf.reduce_sum(tf.stack(a_list, axis=1), axis=2), self.random_gaussians * self.sqrt_dt)
        first_term_z = tf.expand_dims(tf.reduce_sum(a_times_gaussians, axis=1), axis=-1)
        second_term_z = 0.5 * tf.expand_dims(tf.reduce_sum(a_squared_list, axis=1) / self.N_time, axis=-1)
        z = tf.exp(first_term_z + second_term_z)
        # self.test_1 = a_squared_list
        # self.test_1 = trajectories[1]
        # self.test_1 = tf.reduce_sum(z) / self.N_batch_size
        # self.test_1 = trajectories[self.N_time - 1]
        # self.test_2 = trajectories[self.N_time - 1]
        # self.test_2 = tf.reduce_sum(tf.stack(a_list, axis=1), axis=[2, 1]) * self.dt

        # self.test_3 = trajectories

        # self.test_2 = self.weights_list

        if self.params["Diffusion"]["Type"] == "LV":
            trajectories = tf.exp(trajectories)

        return trajectories, z, neural_nets_list

    """
    def generate_neural_nets_list(self):
        neural_nets_list = [None for i in range(self.N_time)]
        for n_time_step in range(self.N_time):
            neural_nets_list[n_time_step] = self.neural_net(self.trajectories[:, :n_time_step + 1],
                                                            self.weights_list[n_time_step],
                                                            self.biases[n_time_step])
        return neural_nets_list
    """


    def generate_payoff_values(self):
        if self.params["Payoff"] == "AutoCall":
            number_phoenix_dates = len(self.params["PayoffChars"]["PhoenixDates"])
            phoenix_autocall_spreads_list = [None for i in range(number_phoenix_dates)]
            phoenix_autocall_stay_spreads_previous_proba_list = [None for i in range(number_phoenix_dates)]
            phoenix_dates = self.params["PayoffChars"]["PhoenixDates"]
            phoenix_autocall_spreads = self.params["PayoffChars"]["PhoenixAutoCallSpreads"]
            phoenix_coupons_chars = self.params["PayoffChars"]["PhoenixCoupons"]
            pdi_chars = self.params["PayoffChars"]["PDI"]

            for i in range(number_phoenix_dates):
                cur_autocall_spread = tf.nn.relu(self.trajectories[phoenix_dates[i]]
                                                 - phoenix_autocall_spreads[i][0]) \
                                      - tf.nn.relu(self.trajectories[phoenix_dates[i]]
                                                   - phoenix_autocall_spreads[i][1])
                cur_autocall_spread *= 1./(phoenix_autocall_spreads[i][1] - phoenix_autocall_spreads[i][0])
                phoenix_autocall_spreads_list[i] = cur_autocall_spread
                if i == 0:
                    phoenix_autocall_stay_spreads_previous_proba_list[i] = np.full((self.N_batch_size, 1), 1.)
                if i > 0:
                    phoenix_autocall_stay_spreads_previous_proba_list[i] = \
                    phoenix_autocall_stay_spreads_previous_proba_list[i - 1] * (1. - cur_autocall_spread)
            phoenix_autocall_spreads = tf.math.multiply(
                tf.reduce_sum(tf.stack(phoenix_autocall_spreads_list, axis=1), axis=2),
                tf.reduce_sum(tf.stack(phoenix_autocall_stay_spreads_previous_proba_list, axis=1), axis=2))
            # phoenix_autocall_spreads = tf.reduce_sum(tf.stack(phoenix_autocall_spreads_list, axis=1), axis=2)
            phoenix_coupons_2D = np.repeat(phoenix_coupons_chars, self.N_batch_size, axis=0)
            phoenix_coupons = tf.expand_dims(tf.reduce_sum(tf.multiply(phoenix_autocall_spreads, phoenix_coupons_2D), axis=-1), axis=-1)

            number_of_puts_in_spread = (1. - pdi_chars[0])/(pdi_chars[1] - pdi_chars[0])
            pdi = (number_of_puts_in_spread + 1.) * tf.nn.relu(pdi_chars[1] - self.trajectories[self.N_time]) \
                  - number_of_puts_in_spread * tf.nn.relu(pdi_chars[0] - self.trajectories[self.N_time])

            proba_of_non_autocalled = tf.multiply(
                phoenix_autocall_stay_spreads_previous_proba_list[number_phoenix_dates - 1],
                (1. - phoenix_autocall_spreads_list[number_phoenix_dates - 1]))

            # proba_of_non_autocalled = tf.expand_dims((1. - tf.reduce_max(phoenix_autocall_spreads, axis=1)), axis=-1)
            return_value = phoenix_coupons - tf.multiply(pdi, proba_of_non_autocalled)
            # self.test_2 = pdi
        elif self.params["Payoff"] == "Calls&Puts":
            return_value = self.number_of_calls * tf.nn.relu(self.trajectories[self.N_time] - self.K) \
                           + self.number_of_puts * tf.nn.relu(self.K_2 - self.trajectories[self.N_time])
        elif self.params["Payoff"] == "Call":
            return_value = tf.nn.relu(self.trajectories[self.N_time] - self.K)
        return return_value

    def generate_price(self):
        payoff_times_z = tf.multiply(self.payoff_values, 1. / self.z)
        price = tf.reduce_sum(payoff_times_z, axis=0) / self.N_batch_size
        return price

    def generate_variance(self):
        payoff_times_z = tf.multiply(self.payoff_values, 1./ self.z)
        payoff_times_z_squared = tf.square(payoff_times_z)
        sum_of_payoff_times_z_squared = tf.reduce_sum(payoff_times_z_squared, axis=[0, 1])
        variance = sum_of_payoff_times_z_squared / self.N_batch_size - tf.square(self.price)
        return variance

    def generate_variance_z(self):
        variance_z = tf.math.reduce_mean(tf.square(self.z))
        return variance_z

    def generate_loss(self):
        loss_value = self.variance + self.lambda_constraint * tf.math.log(1 + tf.nn.relu(self.variance_z - self.constraint))
        # loss_value = self.std + self.lambda_constraint * tf.math.log(1 + tf.nn.relu(self.variance_z - self.constraint))
        return loss_value

    def generate_layers_list(self):
        layers_list = [None for i in range(self.N_time)]
        for n_time_step in range(self.N_time):
            if self.params["a"] == "full":
                layer = self.get_layers(n_time_step + 1)
            elif self.params["a"] == "local":
                layer = self.get_layers(1)
            layers_list[n_time_step] = layer
        return layers_list

    def get_layers(self, number_of_inputs):
        layers = [number_of_inputs] + 2 * [16] + [1]
        return layers

    def generate_weights_list_and_biases_list(self):
        weights_list = [None for i in range(self.N_time)]
        biases_list = [None for i in range(self.N_time)]
        for n_time_step in range(self.N_time):
            if n_time_step == 0:
                test_1 = tf.Variable(tf.zeros([1, 1], dtype=tf.float64))
                # test_1 = tf.constant(0., shape=[1, 1], dtype=tf.float64)
                b = tf.tile(test_1, multiples=[self.N_batch_size, 1])
                biases_list[0] = b
            else:
                weights_list[n_time_step], biases_list[n_time_step] = self.initialize_NN(self.layers_list[n_time_step])
        return weights_list, biases_list

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            # b = self.xavier_init(size=[1, layers[l+1]])
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]


        xavier_stddev = np.sqrt(2/(in_dim + out_dim))

        return_value = tf.Variable(tf.truncated_normal([in_dim, out_dim],
                                                       stddev=xavier_stddev,
                                                       dtype=tf.float64, seed=3),
                                   dtype=tf.float64)
                                   




        """
        return_value = tf.Variable(tf.zeros([in_dim, out_dim], dtype=tf.float64), dtype=tf.float64)
        """


        return return_value

    def neural_net(self, X, weights, biases, t_step=None):
        """
        if not X:
            b_old = tf.Variable(tf.zeros([self.N_batch_size, 1], dtype=tf.float64))
            test_old = tf.Variable(tf.zeros([1, 1], dtype=tf.float64))
            # test = tf.constant(0, dtype=tf.float64, shape=[1, 1])
            # b = tf.tile(test, multiples=[self.N_batch_size, 1])
            test_1 = tf.Variable(tf.zeros[1, 1], dtype=tf.float64)
            b = tf.tile(test_1, multiples=[self.N_batch_size, 1])
            self.test_3 = b
            return b
        """
        if t_step == 0:
            Y = biases
        else:
            num_layers = len(weights) + 1
            H = tf.reduce_sum(tf.stack(X, axis=1), axis=2) - self.x

            for l in range(0, num_layers-2):
                W = weights[l]
                b = biases[l]
                # H = tf.sin(tf.add(tf.matmul(H, W), b))
                # H = tf.tanh(tf.add(tf.matmul(H, W), b))
                # H = tf.sigmoid(tf.add(tf.matmul(H, W), b)) # used to be sigmoid
                H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
                # H = tf.nn.relu(tf.matmul(H, W))
                # H = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
                # H = tf.nn.elu(tf.add(tf.matmul(H, W), b))



            W = weights[-1]
            # b = biases[-1]

            # Y = tf.add(tf.matmul(H, W), b)

            Y = tf.matmul(H, W)

        """
        if t_step == 0:
            self.test_3 = Y
        """
        return Y

    """
    # @profile
    def loss_function(self):
        loss = 0.
        self.test_1 = []
        self.test_2 = []
        self.test_3 = []

        X = tf.concat([self.q_batched,
                       self.N_batch_resized,
                       self.running_costs_batch,
                       self.mu_hats_batch_level_sorted_resized,
                       self.m_level_batch_sorted_resized,
                       self.S_level_batch_sorted_upper_or_diag_resized], 1)

        phi_x = self.neural_net(X, self.weights, self.biases)

        difference = self.level_expectations_batch_resized - phi_x
        power_diff = tf.pow(tf.abs(difference), self.r)
        loss = tf.reduce_sum(power_diff, [0, 1])
        loss /= (self.j_batch_size * self.k_batch_size)

        return loss
    """

    def fetch_minibatch(self):
        random_gaussians_batch = np.random.normal(size=(self.N_batch_size, self.N_time))
        return random_gaussians_batch

    @staticmethod
    def get_conf_and_diffusion_types(conf, diffusion):
        conf_type = ""
        diffusion_type = ""
        if conf == "Calls&PutsSym":
            conf_type = "calls_and_puts_symmetric"
        if conf == "Calls&PutsAsym":
            conf_type = "calls_and_puts_asymmetric"
        elif conf == "Call":
            conf_type = "call"
        elif conf == "AutoCallSingleCoupon":
            conf_type = "autocall_single_coupon"
        elif conf == "AutoCallMultiCoupons":
            conf_type = "autocall_multi_coupons"

        if diffusion == "Bachelier":
            diffusion_type = "bachelier"
        elif diffusion == "LV":
            diffusion_type = "lv"

        return conf_type, diffusion_type

    def eval(self, number_of_batches, doing_robust_graph=False):
        start_time = time.time()
        prices = np.full(number_of_batches, 0.)
        if self.params["General"]["Run"]["DoWeightsGraph"]:
            z = np.full((number_of_batches, self.N_batch_size, 1), 0.)
        if self.params["General"]["Run"]["DoTrajectoriesGraph"]:
            trajectories = np.full((number_of_batches, self.N_batch_size, 1), 0.)
        std = np.full(number_of_batches, 0.)
        test_2 = [None for i in range(number_of_batches)]
        for batch_number in range(number_of_batches):
            start_batch_number_time = time.time()
            random_gaussians_batch = self.fetch_minibatch()

            if self.params["Diffusion"]["Type"] == "Bachelier":
                tf_dict = {
                    self.learning_rate: 1,
                    self.random_gaussians: random_gaussians_batch,
                    self.x: self.params["Diffusion"]["X"],
                    self.Sigma: self.params["Diffusion"]["Sigma"]
                }
            elif self.params["Diffusion"]["Type"] == "LV":
                tf_dict = {
                    self.learning_rate: 1,
                    self.random_gaussians: random_gaussians_batch,
                    self.x: self.params["Diffusion"]["X"],
                    self.A: self.params["Diffusion"]["A"],
                    self.B: self.params["Diffusion"]["B"],
                    self.Rho: self.params["Diffusion"]["Rho"],
                    self.M: self.params["Diffusion"]["M"],
                    self.Sigma: self.params["Diffusion"]["Sigma"]
                }
            if not doing_robust_graph:
                if self.params["General"]["Run"]["DoTrajectoriesGraph"] and (not self.params["General"]["Run"]["DoWeightsGraph"]):
                    prices[batch_number], std[batch_number], test_2[batch_number], trajectories[batch_number, :, :] = self.sess.run([self.price, self.std, self.test_2, self.trajectories[self.N_time]], tf_dict)
                if self.params["General"]["Run"]["DoWeightsGraph"] and (not self.params["General"]["Run"]["DoTrajectoriesGraph"]):
                    prices[batch_number], std[batch_number], test_2[batch_number], z[batch_number, :, :] = self.sess.run([self.price, self.std, self.test_2, self.z], tf_dict)
                if self.params["General"]["Run"]["DoWeightsGraph"] and self.params["General"]["Run"]["DoTrajectoriesGraph"]:
                    prices[batch_number], std[batch_number], test_2[batch_number], trajectories[batch_number, :, :], z[batch_number, :, :] = self.sess.run([self.price, self.std, self.test_2, self.trajectories[self.N_time], self.z], tf_dict)
                if ((not self.params["General"]["Run"]["DoTrajectoriesGraph"]) and (not self.params["General"]["Run"]["DoWeightsGraph"])):
                    prices[batch_number], std[batch_number], test_2[batch_number] = self.sess.run([self.price, self.std, self.test_2], tf_dict)
                end_batch_number_time = time.time()
                if batch_number == 0: print("Time for 1 eval iteration:", end_batch_number_time - start_batch_number_time)
            else:
                prices[batch_number], std[batch_number], test_2[batch_number] = self.sess.run(
                    [self.price, self.std, self.test_2], tf_dict)


        # print("test 2 = ", test_2)

        if ((not doing_robust_graph) and self.params["General"]["Run"]["DoWeightsGraph"]):
            z = z.flatten()
            fig = plt.figure()
            n, bins, patches = plt.hist(z, 500, density=True, facecolor="b", alpha=0.75, range=(0., 10.))
            # plt.xlabel(r"$\bar Z^{\hat \theta}_T$")
            # plt.ylabel("Probability")
            # plt.title(r"Distribution Function of $\bar Z^{\hat \theta}_T$")
            if self.params["General"]["Run"]["ShowWeightsGraph"]:
                plt.show()
            conf_type, diffusion_type = self.get_conf_and_diffusion_types(self.params["Conf"], self.params["Diffusion"]["Type"])
            a_type = "a_" + self.params["a"]
            file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_weight_" + conf_type + "_" + diffusion_type + ".png"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(file_path)
            plt.close()

            fig = plt.figure()
            n, bins, patches = plt.hist(z, 500, density=True, facecolor="b", alpha=0.75, range=(0.1, 10.))
            logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            plt.close()
            fig = plt.figure()
            plt.hist(z, bins=logbins, density=True, facecolor="b", alpha=0.75)
            # plt.xlabel(r"$\bar Z^{\hat \theta}_T$")
            plt.xscale('log')
            # plt.ylabel("Probability")
            # plt.title(r"Distribution Function of $\bar Z^{\hat \theta}_T$ in Log Scale")
            if self.params["General"]["Run"]["ShowWeightsGraph"]:
                plt.show()
            conf_type, diffusion_type = self.get_conf_and_diffusion_types(self.params["Conf"], self.params["Diffusion"]["Type"])
            a_type = "a_" + self.params["a"]
            file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_weight_log_scale_" + conf_type + "_" + diffusion_type + ".png"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(file_path)
            plt.close()

            file_path_json = "./figures/" + conf_type + "/" + diffusion_type + "/json_weight_" + conf_type + "_" + diffusion_type + ".json"
            params_json = json.dumps(self.params, indent=4)
            f = open(file_path_json, "w")
            f.write(params_json)
            f.close()
            plt.close()

        if ((not doing_robust_graph) and self.params["General"]["Run"]["DoTrajectoriesGraph"]):
            trajectories = trajectories.flatten()
            fig = plt.figure()
            n, bins, patches = plt.hist(trajectories, 500, density=True, facecolor="b", alpha=0.75, range=(0.0, 2.0), label=r"Distribution of $\bar X_T$ Under $\mathbb{Q}^{\hat \theta^*}$")
            # plt.xlabel(r"$x$")
            # plt.ylabel("Probability")
            # plt.title(r"Distribution of $\bar X_T$ Under $\mathbb{Q}^{\hat \theta^*}$")


            if self.params["Diffusion"]["Type"] == "Bachelier":
                x_gaussian_range = np.arange(0.0, 2.0, 0.01)
                y_gaussian = scipy.stats.norm.pdf(x_gaussian_range,
                                                  self.params["Diffusion"]["X"],
                                                  self.params["Diffusion"]["Sigma"] * self.params["PayoffChars"]["T"])
                plt.plot(x_gaussian_range, y_gaussian, label=r"Distribution of $\bar X_T$ Under Original $\mathbb{Q}$")

                # plt.subplots_adjust(bottom=0.3)
                # plt.legend(loc='upper center', bbox_to_anchor=(0.5, - 0.2))

            if self.params["General"]["Run"]["ShowTrajectoriesGraph"]:
                plt.show()
            conf_type, diffusion_type = self.get_conf_and_diffusion_types(self.params["Conf"], self.params["Diffusion"]["Type"])
            a_type = "a_" + self.params["a"]
            file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_trajectories_" + conf_type + "_" + diffusion_type + ".png"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(file_path)
            file_path_json = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/json_trajectories_" + conf_type + "_" + diffusion_type + ".json"
            params_json = json.dumps(self.params, indent=4)
            f = open(file_path_json, "w")
            f.write(params_json)
            f.close()
            plt.close()

        average_price = np.average(prices)
        global_variance = np.sum(np.square(std)) * self.N_batch_size
        global_std = np.sqrt(global_variance) / (self.N_batch_size * number_of_batches)
        print("ad price: ", average_price)
        print("ad std: ", global_std)
        end_time = time.time()
        print("Time spent in NeuralNetwork.eval():", end_time - start_time)
        return average_price, global_std

    def train(self, number_of_batches, learning_rate):
        print("Starting NeuralNetwork.train()")
        start_time_train = time.time()
        start_time_N_iter = time.time()

        N_Iter = number_of_batches * self.N_batch_change_proportion
        test_1 = 0.
        for it in range(N_Iter):
            if it % self.N_batch_change_proportion == 0:
                random_gaussians_batch = self.fetch_minibatch()

                if self.params["Diffusion"]["Type"] == "Bachelier":
                    tf_dict = {
                        self.learning_rate: learning_rate,
                        self.random_gaussians: random_gaussians_batch,
                        self.x: self.params["Diffusion"]["X"],
                        self.Sigma: self.params["Diffusion"]["Sigma"]
                    }
                elif self.params["Diffusion"]["Type"] == "LV":
                    tf_dict = {
                        self.learning_rate: learning_rate,
                        self.random_gaussians: random_gaussians_batch,
                        self.x: self.params["Diffusion"]["X"],
                        self.A: self.params["Diffusion"]["A"],
                        self.B: self.params["Diffusion"]["B"],
                        self.M: self.params["Diffusion"]["M"],
                        self.Rho: self.params["Diffusion"]["Rho"],
                        self.Sigma: self.params["Diffusion"]["Sigma"]
                    }


            """
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            self.sess.run(self.train_op, tf_dict, options=run_options, run_metadata=run_metadata)
            """


            self.sess.run(self.train_op, tf_dict)

            """
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
            """

            if it % self.N_iter_show_proportion == 0:
                loss, std, price, test_1, test_2, test_3 = self.sess.run([self.loss, self.std, self.price, self.test_1, self.test_2, self.test_3], tf_dict)
                real_call_price = self.bachelier_call_price()
                print("It: ", it,
                      "real_price: ", real_call_price,
                      "price: ", price,
                      "std: ", std / np.sqrt(self.N_batch_size),
                      "loss: ", loss,
                      "test_1: ", test_1,
                      "test_2: ", test_2,
                      "test_3: ", test_3,
                      "Time for N_Iter iterations: ", time.time() - start_time_N_iter)
                start_time_N_iter = time.time()
        end_time_train = time.time()
        print("Time spent in NeuralNetwork.train(): ", end_time_train - start_time_train)
        print("End of NeuralNetwork.train")
        return test_1
