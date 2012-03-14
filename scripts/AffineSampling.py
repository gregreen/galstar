#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       AffineSampling.py
#       
#       Copyright 2012 Greg <greg@greg-G53JW>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#       


import numpy as np
from scipy.special import erf, gamma, gammainc
from math import log, sqrt, pi
import random

from kmeans import k_means_affine


class AffineSampler:
	'''An affine sampler, using the ensemble method developed in Goodman & Weare (2010), and also implemented recently by Foreman-Mackey, Hogg, Lang et al. (2012).'''
	
	# The chain
	chain = None			# The full chain, in the form [x_0, x_1, ...]
	weight = None			# Multiplicity of each element of the chain
	pi_chain = None			# Unnormalized posterior for each element in the chain
	N = None				# Dimensionality of parameter space
	
	# Information on the acceptance rate
	N_accepted = None
	N_rejected = None
	
	# Ensemble of states
	X = None				# Ensemble of states X = (x_0, x_1, ..., x_L)
	X_weight = None			# Number of steps x_j has spent on current point
	pi_X = None				# p(x_j) for each j
	L = None				# Number of samplers in ensemble (L > n)
	
	# Proposal points (one for each sampler in ensemble)
	Y = None
	pi_Y = None
	accepted = None
	
	# Stretch size
	a = None
	sqrt_a = None
	
	# Input functions and parameters
	f_rand_state = None		# Generates a random state
	f_logpdf = None			# Evaluates log p(X)
	params = None			# Extra parameters needed to evaluate logpdf
	
	def __init__(self, f_rand_state, f_logpdf, params, a=2., L=None):
		'''Initialize the affine sampler
			f_rand_state: function which generates a random point
			f_logpdf: function which returns log p(X), up to a normalizing constant
			params: any parameters needed by f_logpdf
			a: Dimensionless scale of stretch move. Must be greater than unity.
			L: Number of samplers in ensemble. Must be greater than dimensionality of parameter space.'''
		self.f_rand_state = f_rand_state
		self.f_logpdf = f_logpdf
		self.params = params
		# Initialize chain
		self.chain = []
		self.weight = []
		self.pi_chain = []
		self.N = len(f_rand_state())
		# Initialize the chain statistics
		self.N_accepted = 0
		self.N_rejected = 0
		# Initialize the ensemble
		self.L = L
		if self.L == None:
			self.L = 10 * N		# 10 samplers per dimension are probably the minimum one wants
		self.X, self.Y = [], []
		self.pi_X, self.pi_Y = np.empty(self.L, dtype=np.float), np.empty(self.L, dtype=np.float)
		self.X_weight = np.empty(self.L, dtype=int)
		self.accepted = np.empty(self.L, dtype=bool)
		for i in range(L):
			self.X.append(f_rand_state())
			self.Y.append(f_rand_state())
			self.pi_X[i] = f_logpdf(self.X[-1], self.params)
			self.pi_Y[i] = 0
			self.X_weight[i] = 1
		# Set the stretch size
		self.set_scale(a)
	
	def set_scale(a):
		'''Set the dimensionless sampling scale'''
		self.a = a
		self.sqrt_a = sqrt(a)
	
	def Z_stretch(self):
		return ((self.sqrt_a - 1./self.sqrt_a) * random.random() + 1./self.sqrt_a)**2.
	
	def draw(self, j):
		'''Generate a proposal point, and return the stretch size Z'''
		# Choose the point to stretch past
		k = random.randint(0, self.L-2)
		if k >= j:
			k += 1
		# Generate Y by a stretch move
		Z_tmp = self.Z_stretch()
		self.Y[j] = self.X[k] + Z_tmp * (self.X[j] - self.X[k])
		self.pi_Y[j] = self.f_logpdf(self.Y[j], self.params)
		# Return the stretch size
		return Z_tmp
	
	def accept(self, j):
		'''Accept the proposal point'''
		# Log the old point x_j
		#if j == 0:
		self.chain.append(self.X[j])
		self.weight.append(self.X_weight[j])
		self.pi_chain.append(self.pi_X[j])
		self.N_accepted += 1
		# Update x_j with the proposal y_j
		self.X[j] = self.Y[j]
		self.pi_X[j] = self.pi_Y[j]
		self.X_weight[j] = 1
	
	def step(self):
		'''Take one Metropolis-Hastings step for each sampler in ensemble'''
		self.accepted.fill(False)
		# Propose a new point for each X in the ensemble and determine whether to accept
		for j in range(self.L):
			Z_tmp = self.draw(j)		# Draw a proposal
			log_P_accept = (self.N - 1.) * log(Z_tmp) + self.pi_Y[j] - self.pi_X[j]	# Determine acceptance probability
			if log_P_accept >= 0:	# Accept immediately if P_accept >= 1
				self.accepted[j] = True
			elif log(random.random()) < log_P_accept:	# Else accept with probability Z^(n-1) p(Y) / p(X_0)
				self.accepted[j] = True
			else:
				self.N_rejected += 1
				self.X_weight[j] += 1
		# Update the ensemble
		for j in range(self.L):
			if self.accepted[j]:
				self.accept(j)
	
	def reset_chain(self):
		'''Clear the chain'''
		self.chain = []
		self.weight = []
		self.pi_chain = []
		# Set the weight of each current point in the ensemble to unity
		for j in range(self.L):
			self.X_weight[j] = 1
	
	def flush(self):
		'''Add all of the points in the current state to the chain'''
		for j in range(self.L):
			#if j == 0:
			self.chain.append(self.X[j])
			self.weight.append(self.X_weight[j])
			self.pi_chain.append(self.pi_X[j])
			self.X_weight[j] = 0
	
	def update_proposal(self):
		'''Set the proposal covariance to the covariance of the chain'''
		self.cov = self.get_cov()
		eival, eivec = np.linalg.eigh(self.cov)
		sqrteival = np.matrix(np.diagflat(np.sqrt(eival)))
		self.sqrtcov = eivec * sqrteival
	
	def set_scale(self, a):
		'''Set the step size'''
		self.a = a
		self.sqrt_a = sqrt(a)
	
	def get_acceptance(self):
		'''Get the acceptance rate of the chain'''
		return float(self.N_accepted) / float(self.N_accepted + self.N_rejected)
	
	def get_chain(self):
		'''Return a numpy array containing the chain in the form [X_0, X_1, ...]'''
		return np.array(self.chain)
	
	def get_weights(self):
		'''Return a numpy array containing the chain weights in the form [w_0, w_1, ...]'''
		return np.array(self.weight)
	
	def get_mean(self):
		'''Return a numpy array containing the means of the chain'''
		return calc_mean(self.chain, self.weight)
	
	def get_cov(self):
		'''Return a numpy matrix representing the covariance of the chain'''
		return calc_cov(self.chain, self.weight)
	
	def Z_harmonic(self, nsigma=2.):
		'''Return an estimate of the Bayesian evidence Z, using the harmonic mean method'''
		return calc_Z_harmonic(self.chain, self.weight, self.pi_chain, nsigma)
	
	def get_norm_chain(self, k=2, nsigma=2., iterations=10):
		'''Return a normalized chain. Finds k modes in posterior, weighting each according to its evidence.'''
		# Split up chain into k clusters
		length = len(self.chain)
		means_guess = [self.chain[random.randint(0, length-1)] for i in range(k)]
		cluster_mask = k_means_affine(self.chain, means_guess, iterations)
		
		# Create the numpy arrays to return
		chain_arr = np.array(self.chain)
		weight_arr = np.array(self.weight, dtype=np.float)
		pi_arr = np.array(self.pi_chain)
		
		# Add weighted points from each cluster to the chain
		Z_total = 0.
		correction_total = 0.
		for i in range(k):
			x_cluster = chain_arr[cluster_mask[i]]
			weight_cluster = weight_arr[cluster_mask[i]]
			pi_cluster = pi_arr[cluster_mask[i]]
			Z = calc_Z_harmonic(x_cluster, weight_cluster, pi_cluster, nsigma)
			Z_total += Z
			correction = 1. / np.sum(weight_cluster) / Z
			correction_total += correction
			print 'Cluster %d:' % i
			print 'mean: ', x_cluster.mean(0)
			print 'Z = %.3g' % Z
			print 'N = %d' % len(weight_cluster)
			print 'P = %.3g' % (np.sum(weight_cluster) * correction)
			for j in range(length):
				if cluster_mask[i,j]:
					weight_arr[j] *= correction
		
		# Return the chain and weights
		return chain_arr, weight_arr / correction_total


def calc_mean(chain, weight):
	'''Return a numpy array containing the means of the chain'''
	N = len(chain[0])
	mu = np.zeros(N, dtype=np.float)
	N_samples = 0.
	for i in range(len(chain)):
		mu += chain[i] * weight[i]
		N_samples += weight[i]
	return mu / float(N_samples)

def calc_cov(chain, weight):
	'''Return a numpy matrix representing the covariance of the chain'''
	N = len(chain[0])
	chain_arr = np.array(chain)
	mu = calc_mean(chain, weight)
	# Add the square deviation from each element of the chain
	tmp = np.matrix(np.zeros((N, N), dtype=np.float))
	for i in range(len(chain_arr)):
		dx = np.matrix(chain_arr[i] - mu)
		tmp += weight[i] * dx.T * dx
	return tmp / float(sum(weight) - 1)

def calc_Z_harmonic(chain, weight, pi_chain, nsigma=2.):
	'''Return an estimate of the Bayesian evidence Z, using the harmonic mean method'''
	# Get the covariance matrix of the chain
	N = len(chain[0])
	cov = calc_cov(chain, weight)
	invcov = cov.I
	mu = calc_mean(chain, weight)
	sigma_dist2 = lambda x: float(np.matrix(x) * invcov * np.matrix(x).T)
	# Determine the prior volume
	V = sqrt(np.linalg.det(cov)) * (2. * pi**(N/2.) * nsigma**N) / (N * gamma(N/2.))
	# Determine <1/L> inside the prior volume
	tmp = 0.
	total_weight = 0.
	for i in range(len(chain)):
		dist = sigma_dist2(chain[i] - mu)
		if dist < nsigma*nsigma:
			tmp += weight[i] / np.exp(pi_chain[i])
			total_weight += weight[i]
	return V * (total_weight / tmp) * sum(weight)/total_weight



def main():
	
	return 0

if __name__ == '__main__':
	main()

