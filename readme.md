# Deep Importance Sampling

We present a generic path-dependent importance sampling algorithm where the Girsanov induced
change of probability on the path space is represented by a sequence of neural networks taking the
past of the trajectory as an input. At each learning step, the neural networks’ parameters are trained
so as to reduce the variance of the Monte Carlo estimator induced by this change of measure. This
allows for a generic path dependent change of measure which can be used to reduce the variance of any
path-dependent financial payoff. We show in our numerical experiments that for payoffs consisting
of either a call, an asymmetric combination of calls and puts, a symmetric combination of calls and
puts, a multi coupon autocall or a single coupon autocall, we are able to reduce the variance of the
Monte Carlo estimators by factors between 2 and 9. The numerical experiments also show that the
method is very robust to changes in the parameter values, which means that in practice, the training
can be done offline and only updated on a weekly basis.

## Installation

The code runs with:

- Python 3.6.5
- matplotlib 3.1.1
- pandas 0.25.2
- scikit-learn 0.21.3
- numpy 1.16.4
- tensorflow 1.13.1

## Project Structure

The project consists of 3 main files, as well as a params directory. 

- main.pdf is the scientific article also available on arXiv describing the algorithm. 
- main.py contains most of the non tensorflow code, which generates most graphs
- NN.py contains most tensorflow related code, with the construction of neural networks and trajectories
- The params folders contains the parameters used for each specific payoff, volatility parameter and diffusion type, as well as the general run parameters. 