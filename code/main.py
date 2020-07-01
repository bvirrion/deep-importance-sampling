"""
@author: Benjamin Virrion
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy.stats
#import copy
import math
import time
import pandas as pd
from tempfile import TemporaryFile
import random
import tensorflow as tf
# from numba import jit
# import cProfile
from tensorflow.python.client import timeline
from NN import NeuralNetwork
from matplotlib import rc
import matplotlib as mpl
import os
import json
import copy
from mpl_toolkits.mplot3d import Axes3D

# rc('mathtext', fontset="stix")
# rc('text', usetex=True)
# mpl.rcParams["mathtext.fontset"] = "stix"

mpl.rc('text', usetex = True)
mpl.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : [r'\usepackage{amsfonts, amsmath}']}
mpl.pyplot.rcParams.update(params)

def sigma_for_lv_graph(params, x, t):
    log_x = np.log(x)
    a = np.float64(params["Diffusion"]["A"])
    b = np.float64(params["Diffusion"]["B"])
    rho = np.float64(params["Diffusion"]["Rho"])
    m = np.float64(params["Diffusion"]["M"])
    sigma = np.float64(params["Diffusion"]["Sigma"])
    term_in_sqrt_term = np.square(log_x - m) + np.square(sigma)
    sqrt_term = np.sqrt(term_in_sqrt_term)

    partial_t_w = a + b * (rho * (log_x - m) + sqrt_term)
    w = t * partial_t_w
    partial_k_w = t * b * (np.divide(log_x - m, sqrt_term))
    partial_kk_w = np.divide(t * b * np.square(sigma), np.power(term_in_sqrt_term, 1. / 3.))

    denominator_first_term = 1 - log_x / w * partial_k_w
    denominator_second_term = 0.25 * (- 0.25 - 1. / w + np.divide(np.square(log_x), np.square(w))) * np.square(
        partial_k_w)
    denominator_third_term = 0.5 * partial_kk_w

    sigma_square = np.divide(partial_t_w, denominator_first_term + denominator_second_term + denominator_third_term)

    sigma = np.sqrt(sigma_square)

    return sigma

def sigma_for_imp_graph(params, x, t):
    log_x = np.log(x)
    a = np.float64(params["Diffusion"]["A"])
    b = np.float64(params["Diffusion"]["B"])
    rho = np.float64(params["Diffusion"]["Rho"])
    m = np.float64(params["Diffusion"]["M"])
    sigma = np.float64(params["Diffusion"]["Sigma"])
    term_in_sqrt_term = np.square(log_x - m) + np.square(sigma)
    sqrt_term = np.sqrt(term_in_sqrt_term)

    partial_t_w = a + b * (rho * (log_x - m) + sqrt_term)
    w = t * partial_t_w
    partial_k_w = t * b * (np.divide(log_x - m, sqrt_term))
    partial_kk_w = np.divide(t * b * np.square(sigma), np.power(term_in_sqrt_term, 1. / 3.))

    denominator_first_term = 1 - log_x / w * partial_k_w
    denominator_second_term = 0.25 * (- 0.25 - 1. / w + np.divide(np.square(log_x), np.square(w))) * np.square(
        partial_k_w)
    denominator_third_term = 0.5 * partial_kk_w

    sigma_square = np.divide(partial_t_w, denominator_first_term + denominator_second_term + denominator_third_term)

    sigma = np.sqrt(sigma_square)

    return np.sqrt(w / t)

def do_lv_graph(params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0.4, 1.6, 0.1)
    y = np.arange(0.1, 1., 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array(sigma_for_lv_graph(params, np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    # ax.set_xlabel(r'Strike')
    ax.set_xlabel(r'$k$')
    # ax.set_ylabel(r'Maturity')
    ax.set_ylabel(r'$t$')
    # ax.set_zlabel(r'$\sigma$')

    title = "Local Volatility Surface"
    # plt.title(title, pad=20)

    if params["General"]["Run"]["ShowLVGraph"]:
        plt.show()
    file_path = "./figures/local_volatility_surface.png"
    fig.savefig(file_path)
    plt.close()

def do_imp_graph(params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0.4, 1.6, 0.1)
    y = np.arange(0.1, 1., 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array(sigma_for_imp_graph(params, np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    # ax.set_xlabel(r'Strike')
    ax.set_xlabel(r'$k$')
    # ax.set_ylabel(r'Maturity')
    ax.set_ylabel(r'$t$')
    # ax.set_zlabel(r'$\sigma_{imp}$')

    title = r"Implied Volatility Surface"
    # plt.title(title, pad=20)

    if params["General"]["Run"]["ShowImpGraph"]:
        plt.show()
    file_path = "./figures/implied_volatility_surface.png"
    fig.savefig(file_path)
    plt.close()

def do_robust_graphs(params, nn):
    if params["Diffusion"]["Type"] == "LV":
        parameters = ["X",
                      "A",
                      "B",
                      "Rho",
                      "M",
                      "Sigma"]
    elif params["Diffusion"]["Type"] == "Bachelier":
        parameters = ["X",
                      "Sigma"]

    for parameter in parameters:
        base_value = params["Diffusion"][parameter]
        number_of_steps = 20
        step_size = base_value * 0.8 / number_of_steps

        sigma_array_adaptive = np.full(number_of_steps, 0.)
        sigma_array_mc = np.full(number_of_steps, 0.)
        base_values_array = np.full(number_of_steps, 0.)


        ratio = params["General"]["Run"]["RatioRobustGraphs"]

        for step in range(number_of_steps):
            cur_parameter = base_value * 0.6 + step * step_size
            base_values_array[step] = cur_parameter
            if parameter == "X":
                nn.params["Diffusion"]["X"] = cur_parameter
                av_price, glob_std = nn.eval(params["Run"]["NumberOfBatchesForEval"]//ratio, doing_robust_graph=True)
                mc_price, mc_std = nn.monte_carlo_price(params["Run"]["NumberOfBatchesForEval"]//ratio)
                sigma_array_adaptive[step] = glob_std
                sigma_array_mc[step] = mc_std
                nn.params["Diffusion"]["X"] = base_value
                print("Finished step=", step, "of do_robust_graphs with parameter", parameter)

            if parameter == "A":
                nn.params["Diffusion"]["A"] = cur_parameter
                av_price, glob_std = nn.eval(params["Run"]["NumberOfBatchesForEval"]//ratio, doing_robust_graph=True)
                mc_price, mc_std = nn.monte_carlo_price(params["Run"]["NumberOfBatchesForEval"]//ratio)
                sigma_array_adaptive[step] = glob_std
                sigma_array_mc[step] = mc_std
                nn.params["Diffusion"]["A"] = base_value
                print("Finished step=", step, "of do_robust_graphs with parameter", parameter)

            if parameter == "B":
                nn.params["Diffusion"]["B"] = cur_parameter
                av_price, glob_std = nn.eval(params["Run"]["NumberOfBatchesForEval"]//ratio, doing_robust_graph=True)
                mc_price, mc_std = nn.monte_carlo_price(params["Run"]["NumberOfBatchesForEval"]//ratio)
                sigma_array_adaptive[step] = glob_std
                sigma_array_mc[step] = mc_std
                nn.params["Diffusion"]["B"] = base_value
                print("Finished step=", step, "of do_robust_graphs with parameter", parameter)

            if parameter == "M":
                nn.params["Diffusion"]["M"] = cur_parameter
                av_price, glob_std = nn.eval(params["Run"]["NumberOfBatchesForEval"]//ratio, doing_robust_graph=True)
                mc_price, mc_std = nn.monte_carlo_price(params["Run"]["NumberOfBatchesForEval"]//ratio)
                sigma_array_adaptive[step] = glob_std
                sigma_array_mc[step] = mc_std
                nn.params["Diffusion"]["M"] = base_value
                print("Finished step=", step, "of do_robust_graphs with parameter", parameter)

            if parameter == "Rho":
                nn.params["Diffusion"]["Rho"] = cur_parameter
                av_price, glob_std = nn.eval(params["Run"]["NumberOfBatchesForEval"]//ratio, doing_robust_graph=True)
                mc_price, mc_std = nn.monte_carlo_price(params["Run"]["NumberOfBatchesForEval"]//ratio)
                sigma_array_adaptive[step] = glob_std
                sigma_array_mc[step] = mc_std
                nn.params["Diffusion"]["Rho"] = base_value
                print("Finished step=", step, "of do_robust_graphs with parameter", parameter)

            if parameter == "Sigma":
                nn.params["Diffusion"]["Sigma"] = cur_parameter
                av_price, glob_std = nn.eval(params["Run"]["NumberOfBatchesForEval"]//ratio, doing_robust_graph=True)
                mc_price, mc_std = nn.monte_carlo_price(params["Run"]["NumberOfBatchesForEval"]//ratio)
                sigma_array_adaptive[step] = glob_std
                sigma_array_mc[step] = mc_std
                nn.params["Diffusion"]["Sigma"] = base_value
                print("Finished step=", step, "of do_robust_graphs with parameter", parameter)

        fig, ax1 = plt.subplots()
        ax1.plot(base_values_array, sigma_array_mc, color='tab:blue', linestyle=":")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.plot(base_values_array, sigma_array_adaptive, color="tab:red", linestyle="-")
        ax1.set_ylim(bottom=0.0)

        ax2 = ax1.twinx()
        ax2.plot(base_values_array, sigma_array_adaptive / sigma_array_mc, color="tab:orange", linestyle="--")
        ax2.set_ylim(0.0, 1.5)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        if params["Graph"]["ShowGraph"]:
            plt.show()

        if params["Graph"]["Save"]:
            conf_type, diffusion_type = NeuralNetwork.get_conf_and_diffusion_types(params["Conf"], params["Diffusion"]["Type"])
            a_type = "a_" + params["a"]
            file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_" + conf_type + "_" + diffusion_type + "_robust_" + parameter + ".png"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(file_path)
        plt.close()

    return




def do_a_graph(params, a_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_min = params["Grid"]["x_min"]
    x_step_size = params["Grid"]["x_step_size"]
    x_steps = params["Grid"]["x_steps"]
    x_max = x_min + (x_steps - 1) * x_step_size
    n_time_steps = len(a_list)
    T_mat = params["PayoffChars"]["T"]
    t_min = 0.
    t_max = T_mat
    t_steps = params["PayoffChars"]["NTime"]
    t_step_size = T_mat / n_time_steps
    # t = np.arange(t_min, t_max, t_step_size)
    # x = np.arange(x_min, x_max, x_step_size)
    t = np.linspace(t_min, t_max, num=t_steps)
    x = np.linspace(x_min, x_max, num=x_steps)
    X, T = np.meshgrid(x, t)
    a_np = np.array(a_list)
    zs = a_np
    Z = zs.reshape(T.shape)

    ax.plot_surface(X, T, Z)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    # ax.set_zlabel(r'$\tilde a^{\hat \theta^*}$')

    title = r"$\tilde a^{\hat \theta^*}$ Surface"
    # plt.title(title, pad=20)

    if params["General"]["Run"]["ShowEvalAOnGrid"]:
        plt.show()

    conf_type, diffusion_type = NeuralNetwork.get_conf_and_diffusion_types(params["Conf"], params["Diffusion"]["Type"])
    a_type = "a_" + params["a"]
    for angle_step in range(12):
        for azim_step in range(3):
            azim = 30 * (azim_step - 1)
            angle = 30 * angle_step
            file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/a_surface/" + str(azim_step - 1) + "/graph_a_surface_" + conf_type + "_" + diffusion_type + "_" + str(azim) + "_" + str(angle) + ".png"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            ax.view_init(azim, angle)
            fig.savefig(file_path)
    plt.close()



def do_graph(params):
    number_of_steps = params["Graph"]["NumberOfSteps"]
    if params["Payoff"] == "Call" or params["Payoff"] == "Calls&Puts":
        real_prices_array = np.full(number_of_steps, 0.)
    mc_prices_array = np.full(number_of_steps, 0.)
    mc_stds_array = np.full(number_of_steps, 0.)
    ad_prices_array = np.full(number_of_steps, 0.)
    ad_stds_array = np.full(number_of_steps, 0.)
    x_array = np.full(number_of_steps, 0.)
    x_min = params["Graph"]["XMin"]
    step_size = params["Graph"]["StepSize"]
    number_of_batches_for_training = params["Run"]["NumberOfBatchesForTraining"]
    number_of_batches_for_eval = params["Run"]["NumberOfBatchesForEval"]

    nn_list = [None for i in range(number_of_steps)]
    x_original = copy.deepcopy(params["Diffusion"]["X"])
    all_final_losses = np.full(number_of_steps, 0.)
    for i in range(number_of_steps):
        x = x_min + i * step_size
        params["Diffusion"]["X"] = x
        nn_list[i] = NeuralNetwork(params)
        mc_prices_array[i], mc_stds_array[i] = nn_list[i].monte_carlo_price(number_of_batches=number_of_batches_for_eval)
        if params["Run"]["DoAutomaticLearningRate"]:
            variance = np.square(mc_stds_array[i]) * number_of_batches_for_eval * params["NN"]["NBatchSize"]
            log10_variance = np.floor(np.log10(variance))
            learning_rate = params["Run"]["BaseForAutomaticLearningRate"] / np.power(10, log10_variance)
        elif params["Run"]["DoListLearningRates"]:
            learning_rate = params["Run"]["ListLearningRates"][i]
        else:
            learning_rate = params["Run"]["LearningRate"]
        if params["NN"]["DoAutomaticLambdaConstraint"]:
            variance = np.square(mc_stds_array[i]) * number_of_batches_for_eval * params["NN"]["NBatchSize"]
            log10_variance = np.floor(np.log10(variance))
            lambda_constraint = params["NN"]["BaseForAutomaticLambdaConstraint"] * np.power(10, log10_variance)
            nn_list[i].params["NN"]["LambdaConstraint"] = lambda_constraint
            nn_list[i].lambda_constraint = lambda_constraint

        tf.Session().graph.finalize()
        all_final_losses[i] = nn_list[i].train(number_of_batches=number_of_batches_for_training,
                                               learning_rate=learning_rate)
        ad_prices_array[i], ad_stds_array[i] = nn_list[i].eval(number_of_batches=number_of_batches_for_eval)
        print("step_number =", i)
        print("x = ", x)
        print("learning_rate = ", learning_rate)
        print("lambda_constraint = ", lambda_constraint)
        print("mc_price = ", mc_prices_array[i])
        print("ad_price = ", ad_prices_array[i])
        print("mc_std = ", mc_stds_array[i])
        print("ad_std = ", ad_stds_array[i])
        if params["Diffusion"]["Type"] != "LV":
            if params["Payoff"] == "Call" or params["Payoff"] == "Calls&Puts":
                real_prices_array[i] = nn_list[i].bachelier_call_price()
        x_array[i] = x
        nn_list[i].sess.graph.finalize()
        tf.reset_default_graph()
        nn_list[i].sess.close()
        tf.reset_default_graph()
    params["Diffusion"]["X"] = x_original


    fig, ax1 = plt.subplots()
    # ax1.set_xlabel(r'$x_0$')
    if params["Payoff"] == "Call" or params["Payoff"] == "Calls&Puts":
        if params["Diffusion"]["Type"] != "LV":
            ax1.plot(x_array, real_prices_array, label=r'$\textnormal{Bachelier Price}$', color='black')
    ax1.plot(x_array, ad_prices_array, linestyle=':', label=r'$\textnormal{Adaptative Price}$', color='tab:blue')
    """
    ax1.plot(x_array, ad_prices_array + 300 * ad_stds_array, linestyle=':', label=r'$\textnormal{Adaptative + 300 std}$',
             color='tab:cyan')
    ax1.plot(x_array, ad_prices_array - 300 * ad_stds_array, linestyle=':', label=r'$\textnormal{Adaptative - 300 std}$',
             color='tab:cyan')
    """
    ax1.plot(x_array, mc_prices_array, linestyle='-', label=r'$\textnormal{MC Price}$', color='tab:red')
    """
    ax1.plot(x_array, mc_prices_array + 300 * mc_stds_array, linestyle=':', label=r'$\textnormal{MC + 300 std}$',
             color='tab:pink')
    ax1.plot(x_array, mc_prices_array - 300 * mc_stds_array, linestyle=':', label=r'$\textnormal{MC - 300 std}$',
             color='tab:pink')
    """
    ax1.tick_params(axis='y')
    if params["Diffusion"]["Type"] == "Bachelier":
        my_title = r'Bachelier Prices vs Adaptative Prices vs MC Prices'
    elif params["Diffusion"]["Type"] == "LV":
        my_title = r'Adaptative Prices vs MC Prices for Local Volatility Diffusion'
    # plt.title(my_title)
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.3)
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, - 0.2))

    if params["Graph"]["ShowGraph"]:
        plt.show()

    if params["Graph"]["Save"]:
        conf_type, diffusion_type = NeuralNetwork.get_conf_and_diffusion_types(params["Conf"], params["Diffusion"]["Type"])
        a_type = "a_" + params["a"]
        file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_" + conf_type + "_" + diffusion_type + ".png"
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(file_path)
        file_path_json = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/json_" + conf_type + "_" + diffusion_type + ".json"
        params_json = json.dumps(params, indent=4)
        f = open(file_path_json, "w")
        f.write(params_json)
        f.close()
    plt.close()

    fig, ax1 = plt.subplots()
    # ax1.set_xlabel(r'$x_0$')
    """
    if params["Payoff"] == "Call" or params["Payoff"] == "Calls&Puts":
        error_adaptative = ad_prices_array - real_prices_array
        error_mc = mc_prices_array - real_prices_array
    """

    handle_list = [None for i in range(3)]
    label_list = [None for i in range(3)]
    # ax1.plot(x_array, error_adaptative, label=r'$\textnormal{Adaptative Error}$', color='tab:blue')
    # ax1.plot(x_array, error_mc, label=r'$\textnormal{MC Error}$', color='tab:red')
    handle_list[0], = ax1.plot(x_array, ad_stds_array, label=r'$\textnormal{Adaptative Standard Deviation}$', color='tab:blue', linestyle=":")
    label_list[0] = r'$\textnormal{Adaptative Standard Deviation}$'

    handle_list[1], = ax1.plot(x_array, mc_stds_array, label=r'$\textnormal{MC Standard Deviataion}$', color='tab:red', linestyle="-")
    label_list[1] = r'$\textnormal{MC Standard Deviataion}$'

    # ax1.set_ylabel(r"Standard Deviation")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    my_title = r'Adaptative vs MC Errors and Standard Deviations'
    # plt.title(my_title)
    # plt.tight_layout()
    # ax1.legend(loc=2)

    ax2 = ax1.twinx()
    handle_list[2], = ax2.plot(x_array, ad_stds_array / mc_stds_array, label=r'Ratio of Adaptive and MC Standard Deviations', color="tab:orange", linestyle="--")
    label_list[2] = r'Ratio of Adaptive and MC Standard Deviations'
    # ax2.legend(loc=1)
    # ax2.set_ylabel(r"Ratio")
    ax2.set_ylim(0.0, 1.5)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # plt.subplots_adjust(bottom=0.3)
    # plt.legend(handles=handle_list, labels=label_list, loc='upper center', bbox_to_anchor=(0.5, - 0.2))

    if params["Graph"]["ShowGraph"]:
        plt.show()

    if params["Graph"]["Save"]:
        conf_type, diffusion_type = NeuralNetwork.get_conf_and_diffusion_types(params["Conf"], params["Diffusion"]["Type"])
        a_type = "a_" + params["a"]
        file_path = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_" + conf_type + "_" + diffusion_type + "_errors_and_stds.png"
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(file_path)
    plt.close()

    file_path_all_final_losses = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_" + conf_type + "_" + diffusion_type + "_all_final_losses.txt"
    f = open(file_path_all_final_losses, "w")
    file_string = ""
    for i in range(len(all_final_losses)):
        file_string += "var(Z) for step i = " + str(i) + " is: " + str(all_final_losses[i]) + "\n"
    f.write(file_string)
    f.close()

def do_full_run():
    with open('./params/full_run.json') as json_file:
        params_full_run = json.load(json_file)

    params_dict = {}
    for key in params_full_run.keys():

        # Load graph file
        with open('./params/' + params_full_run[key]["GraphJsonFileName"]) as json_file:
            params_dict[key] = json.load(json_file)

        # Load equivalent of general.json file
        params_dict[key]["General"] = {}
        params_dict[key]["General"]["Run"] = params_full_run[key]

        # Load a.json file
        with open('./params/a.json') as json_file:
            params_dict[key]["a"] = json.load(json_file)["a"]
            if params_dict[key]["General"]["Run"]["AType"] == "local":
                params_dict[key]["a"] = "local"
            elif params_dict[key]["General"]["Run"]["AType"] == "full":
                params_dict[key]["a"] = "full"

        # load diffusion file
        if params_dict[key]["General"]["Run"]["DiffusionType"] == "bachelier":
            with open('./params/diffusion_bachelier.json') as json_file:
                params_dict[key]["Diffusion"] = json.load(json_file)
        elif params_dict[key]["General"]["Run"]["DiffusionType"] == "lv":
            with open('./params/diffusion_lv.json') as json_file:
                params_dict[key]["Diffusion"] = json.load(json_file)

        with open('./params/grid.json') as json_file:
            params_dict[key]["Grid"] = json.load(json_file)

    for key in params_dict.keys():
        print("Doing run for key: ", key)
        if params_dict[key]["General"]["Run"]["DoGraph"]:
            params_temp = copy.deepcopy(params_dict[key])
            params_temp["General"]["Run"]["DoWeightsGraph"] = False
            params_temp["General"]["Run"]["DoTrajectoriesGraph"] = False
            do_graph(params_temp)

        if params_dict[key]["General"]["Run"]["DoSingleRun"]:
            params_single_run = copy.deepcopy(params_dict[key])
            params_single_run["Run"]["NumberOfBatchesForTraining"] *= params_dict[key]["General"]["Run"]["RatioNumberOfBatchesForTraining"]
            params_single_run["Run"]["NumberOfBatchesForEval"] *= params_dict[key]["General"]["Run"]["RatioNumberOfBatchesForEval"]
            do_single_run(params_single_run)


def do_single_run(params):
    nn = NeuralNetwork(params)
    final_loss = nn.train(number_of_batches=params["Run"]["NumberOfBatchesForTraining"],
                          learning_rate=params["Run"]["LearningRate"])
    if params["General"]["Run"]["DoEvalAOnGrid"]:
        a_list = nn.eval_a_on_grid(params["Grid"]["x_min"],
                                   params["Grid"]["x_steps"],
                                   params["Grid"]["x_step_size"])

        do_a_graph(params, a_list)

    monte_carlo_price, monte_carlo_std = nn.monte_carlo_price(number_of_batches=params["Run"]["NumberOfBatchesForEval"])
    print("real price: ", nn.bachelier_call_price())
    print("monte_carlo_price: ", monte_carlo_price)
    print("monte_carlo_std: ", monte_carlo_std)
    _, _, = nn.eval(number_of_batches=params["Run"]["NumberOfBatchesForEval"])

    if params["General"]["Run"]["DoRobustGraphs"]:
        params["General"]["Run"]["RatioRobustGraphs"] = params["General"]["Run"]["RatioNumberOfBatchesForEval"] * params["General"]["Run"]["RatioNumberOfBatchesForTraining"]
        do_robust_graphs(params, nn)

    nn.sess.graph.finalize()
    tf.reset_default_graph()
    nn.sess.close()
    tf.reset_default_graph()

    conf_type, diffusion_type = NeuralNetwork.get_conf_and_diffusion_types(params["Conf"], params["Diffusion"]["Type"])
    a_type = "a_" + params["a"]
    file_path_final_loss = "./figures/" + conf_type + "/" + diffusion_type + "/" + a_type + "/graph_" + conf_type + "_" + diffusion_type + "_final_loss.txt"
    f = open(file_path_final_loss, "w")
    file_string = "Var(Z) for single run is: " + str(final_loss)
    f.write(file_string)
    f.close()

if __name__ == "__main__":
    print("hello world")
    np.random.seed(seed=3)

    with open('./params/graph_bachelier_call.json') as json_file:
        params = json.load(json_file)

    with open('./params/a.json') as json_file:
        params["a"] = json.load(json_file)["a"]
    with open('./params/diffusion_bachelier.json') as json_file:
        params["Diffusion"] = json.load(json_file)
    with open('./params/general.json') as json_file:
        params["General"] = json.load(json_file)

    with open('./params/grid.json') as json_file:
        params["Grid"] = json.load(json_file)

    if params["General"]["Run"]["DoFullRun"]:
        do_full_run()

    if params["General"]["Run"]["DoLVGraph"]:
        do_lv_graph(params)
    if params["General"]["Run"]["DoImpGraph"]:
        do_imp_graph(params)

    if params["General"]["Run"]["DoGraph"]:
        do_graph(params)

    if params["General"]["Run"]["DoSingleRun"]:
        params["Run"]["NumberOfBatchesForTraining"] *= params["General"]["Run"]["RatioNumberOfBatchesForTraining"]
        params["Run"]["NumberOfBatchesForEval"] *= params["General"]["Run"]["RatioNumberOfBatchesForEval"]
        do_single_run(params)




