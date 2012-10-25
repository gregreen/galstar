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


class TMCMC:
	'''
	Markov-Chain Monte Carlo (MCMC) sampler, which draws random samples
	from a target distribution. This sampler implements several different
	step methods, including Metropolis-Hastings (MH) and affine-invariant
	stretch and replacement moves (Goodman & Weare 2010).
	'''
	
	def __init__(self, f_lnp, f_rand_state, cov, size, *args, **kwargs):
		'''
		Inputs:
		    f_lnp         Function or object with __call__() method which
		                  returns ln(p), where p is the target density.
		                  Must be able to take sets of vectors of the form
		                  X[n,dim], where n label the individual vectors,
		                  and dim is the number of dimensions in parameter
		                  space.
		    
		    f_rand_state  Function or object with __call__(n) method which
		                  generates n random state vectors, indexed as
		                  X[n,dim].
		    
		    cov           Initial step covariance, used to generate MH
		                  proposals.
		    
		    size          # of states in ensemble.
		    
		    *args         Arguments to pass to f_lnp.
		    **kwargs      Keyword arguments to pass to f_lnp.
		'''
		self.f_lnp = f_lnp
		self.f_rand_state = f_rand_state
		self.cov = np.array(cov)
		self.size = size
		self.args = args
		self.kwargs = kwargs
		
		# Initialize ensemble of states
		self.X = f_rand_state(self.size)
		self.lnp_X = f_lnp(self.X, self.args, self.kwargs)
		self.weight_X = np.zeros(self.size, dtype='i8')
		self.ndim = self.X.shape[1]
		self.dtype = self.X.dtype
		
		# Proposal states
		self.Y = np.empty((self.size, self.ndim), dtype=self.dtype)
		self.lnp_Y = np.empty(self.size, dtype='f8')
		self.lnp_accept = np.empty(self.size, dtype='f8')
		
		# Chain
		self.chain = []
		self.weight = []
		self.lnp = []
		
		# Statistics
		self.E_k = np.zeros(self.ndim, dtype='f8')
		self.E_ij = np.zeros((self.ndim, self.ndim), dtype='f8')
		self.sum_weight = 0
		
		# Diagnostics
		self.N_accepted = 0
		self.N_rejected = 0
		
		# Affine sampling parameters
		self.a = 2.
		self.sqrt_a = np.sqrt(self.a)
		self.sqrt_Z = np.empty(self.size, dtype='f8')
		
		# Step-type probabilities (the balance is made up by MH steps)
		self.p_stretch = 0.5
	
	#
	# Proposals
	#
	
	def draw_MH(self, j):
		'''
		Propose a Metropolis-Hastings step.
		
		Inputs:
		    j  indices of states in ensemble for which to draw proposals.
		'''
		self.Y[j] = (self.X[j]
		             + np.random.multivariate_normal(np.zeros(self.ndim),
		                                             self.cov,
		                                             size=len(j)))
		self.lnp_Y[j] = self.f_lnp(self.Y[j], self.args, self.kwargs)
	
	def draw_stretch(self, j):
		'''
		Propose an affine-invariant stretch step.
		
		Inputs:
		    j  indices of states in ensemble for which to draw proposals.
		'''
		# Choose point to stretch past
		k = np.random.randint(0, self.size-2, size=len(j))
		idx = (k >= j)
		k[idx] += 1
		# Generate Y by a stretch move
		self.sqrt_Z[j] = ((self.sqrt_a - 1./self.sqrt_a) * np.random.random(len(j))
		                                               + 1./self.sqrt_a)
		self.Y[j] = self.X[k]
		self.Y[j] += np.einsum('n,ni->ni',
		                       self.sqrt_Z[j]*self.sqrt_Z [j],
		                       self.X[j] - self.X[k])
		self.lnp_Y[j] = self.f_lnp(self.Y[j], self.args, self.kwargs)
	
	#
	# Acceptance probabilities
	#
	
	def set_lnp_accept_MH(self, j):
		'''
		Detrmine the acceptance probability for a Metropolis-Hastings
		proposal.
		
		Inputs:
		    j  indices of states in ensemble for which to determine
		       acceptance probabilities.
		'''
		self.lnp_accept[j] = self.lnp_Y[j] - self.lnp_X[j]
	
	def set_lnp_accept_stretch(self, j):
		'''
		Detrmine the acceptance probability for an affine-invariant
		stretch move.
		
		Inputs:
		    j  indices of states in ensemble for which to determine
		       acceptance probabilities.
		'''
		self.lnp_accept[j] = ((self.ndim-1.) * 2.*np.log(self.sqrt_Z[j])
		                                + self.lnp_Y[j] - self.lnp_X[j])
	
	#
	# Stepping procedure
	#
	
	def accept(self, j):
		'''
		Accept proposal states j and update chain and state of ensemble.
		'''
		# Update chain
		self.chain.append(self.X[j])
		self.weight.append(self.weight_X[j])
		self.lnp.append(self.lnp_X[j])
		
		# Update statistics
		self.E_k += np.einsum('n,ni->i', self.weight_X[j], self.X[j])
		self.E_ij += np.einsum('n,ni,nj->ij', self.weight_X[j],
		                                           self.X[j], self.X[j])
		self.sum_weight += np.sum(self.weight_X[j])
		
		# Update state
		self.X[j] = self.Y[j]
		self.lnp_X[j] = self.lnp_Y[j]
		self.weight_X[j] = 1
		self.weight_X[~j] += 1
		
		# Update diagnostics
		N_tmp = np.sum(j)
		self.N_accepted += N_tmp
		self.N_rejected += self.size - N_tmp
	
	def flush(self):
		'''
		Add current state to chain and clear weights of current state.
		'''
		j = self.weight_X > 0
		
		# Update chain
		self.chain.append(self.X[j])
		self.weight.append(self.weight_X[j])
		self.lnp.append(self.lnp_X[j])
		
		# Update statistics
		self.E_k += np.einsum('n,ni->i', self.weight_X[j], self.X[j])
		self.E_ij += np.einsum('n,ni,nj->ij', self.weight_X[j],
		                                           self.X[j], self.X[j])
		self.sum_weight += np.sum(self.weight_X[j])
		
		self.weight_X[j] = 0
	
	def step(self):
		'''
		Take one step for each sampler in ensemble.
		'''
		# Determine which type of step to make for each sampler
		p = np.random.random(size=self.size)
		
		idx = np.where(p < self.p_stretch)[0]
		self.draw_stretch(idx)
		self.set_lnp_accept_stretch(idx)
		p -= self.p_stretch
		
		idx = np.where(p >= 0.)[0]
		self.draw_MH(idx)
		self.set_lnp_accept_MH(idx)
		
		# Accept
		lnp_tmp = np.log(np.random.random(size=self.size))
		idx = (lnp_tmp < self.lnp_accept)
		self.accept(idx)
	
	#
	# Accessors
	#
	
	def get_cov(self):
		'''
		Returns the covariance matrix of the chain.
		'''
		return (self.E_ij / float(self.sum_weight) - 
		           np.outer(self.E_k, self.E_k) / float(self.sum_weight)**2.)
	
	def get_mean(self):
		'''
		Returns the mean of the chain.
		'''
		return self.E_k / float(self.sum_weight)
	
	def get_acceptance(self):
		'''
		Returns the acceptance rate of the chain.
		'''
		return float(self.N_accepted) / float(self.N_accepted+self.N_rejected)
	
	def get_chain(self):
		'''
		Returns the chain and weights.
		
		Outputs:
		    chain   [n,dim] array, where n denotes the order of the state
		            in the chain, and dim denotes the dimension in
		            parameter space.
		    
		    weight  1D array, containing the weight of each state in the
		            chain.
		'''
		self.chain = np.vstack(self.chain)
		self.weight = np.hstack(self.weight)
		return self.chain, self.weight
	
	def get_weight(self):
		'''
		Returns the weight of each state in the chain.
		'''
		self.weight = np.hstack(self.weight)
		return self.weight
	
	def get_lnp(self):
		'''
		Returns the natural log of the probability (unnormalized) of
		each state in the chain.
		'''
		self.lnp = np.hstack(self.lnp)
		return self.lnp
	
	def get_transformed_stats(self, f, *args, **kwargs):
		'''
		Returns the mean and covariance of a transformed chain.
		
		Inputs:
		    f         Transformation to apply to chain. Must take the
		              entire chain in the form chain[n,dim] and output
		              an array of the same dimensions.
		    
		    *args     Arguments to pass to f, other than the chain.
		    
		    **kwargs  Keyword arguments to pass to f.
		
		Outputs:
		    mean  Mean of the transformed chain.
		    
		    cov   Covariance matrix of the transformed chain.
		'''
		self.chain = np.vstack(self.chain)
		self.weight = np.hstack(self.weight)
		chain_transf = f(self.chain, *args, **kwargs)
		mean = np.einsum('n,ni->i', self.weight, chain_transf) / np.sum(self.weight)
		Delta = chain_transf - mean
		cov = np.einsum('n,ni,nj->ij', self.weight, Delta, Delta) / np.sum(self.weight)
		return mean, cov
	
	def find_connected_point(self, nsigma=1., iterations=4, verbose=False):
		'''
		Returns a coordinate where the chain has high density. First,
		a random point is drawn from the chain. This point is already
		likely to lie in a region of high density. Then, a sphere is
		laid down around the current point, and the mean of all points
		in the chain lying within the sphere is computed. This is taken
		to be the new center. This process is iterated a set number of
		times. With each iteration, the radius of the sphere shrinks by
		10%.
		
		Inputs:
		    nsigma      Radius of bounding sphere, in standard
		                deviations. This number will be scaled by
		                sqrt(ndim) (default: 1.0).
		    
		    iterations  # of steps to take (default: 4).
		    
		    verbose     Enables verbose output (default: False).
		'''
		nsigma *= np.sqrt(self.ndim)
		
		invcov = np.linalg.inv(self.get_cov())
		chain, weight = self.get_chain()
		
		# Choose a random point in chain as center
		N = chain.shape[0]
		center = chain[np.random.randint(N)]
		if verbose:
			print 'Starting point:'
			print center
		
		for i in range(iterations):
			# Get distance of each point from from center
			Delta = chain - center
			dist2 = np.einsum('ni,ij,nj->n', Delta, invcov, Delta)
			
			# Set mean of nearby points as center
			idx = (dist2 <= nsigma*nsigma)
			center = np.einsum('i,ij->j', weight[idx], chain[idx]) / np.sum(weight[idx])
			
			if verbose:
				print 'Iteration #%d:' % (i+1)
				print center
			
			nsigma *= 0.9
		
		return center
	
	def get_Z_harmonic(self, nsigma=1.0, frac=0.25, use_mean=False,
	                                                     verbose=False):
		'''
		Returns an estimate of the Bayesian evidence Z, using the
		bounded harmonic mean method. Proceed with extreme caution.
		
		Inputs:
		    nsigma    Radius of sphere, in standard deviations, from
		              which points in the chain are to be drawn. This
		              number will be scaled by sqrt(ndim)
		              (default: 1.0).
		    
		    frac      Maximum fraction of points in chain to use. The
		              volume parameterized by nsigma will be shrunk
		              until this condition is met (default: 0.25).
		    
		    use_mean  Use the mean of the chain, rather than a randomly
		              chosen dense position in the chain, as the
		              center about which the bounding sphere will be
		              drawn (default: False).
		    
		    verbose   Enables verbose output (default: False).
		'''
		# Determine <1/p> inside the prior volume
		cov = self.get_cov()
		invcov = np.linalg.inv(cov)
		chain, weight = self.get_chain()
		lnp = self.get_lnp()
		
		center = None
		if use_mean:
			center = self.get_mean()
		else:
			center = self.find_connected_point(nsigma, verbose=verbose)
		Delta = chain - center
		dist2 = np.einsum('ni,ij,nj->n', Delta, invcov, Delta)
		
		nsigma *= np.sqrt(self.ndim)
		idx = (dist2 < nsigma*nsigma)
		if np.sum(idx) > frac * chain.shape[0]:
			idx = np.argsort(dist2)
			idx = idx[:int(frac * chain.shape[0])]
			tmp = nsigma
			nsigma = np.sqrt(dist2[idx[-1]])
		sum_inv_p = np.dot(weight[idx], np.exp(-lnp[idx]))
		
		# Determine the prior volume
		V = (
		        np.sqrt(np.linalg.det(cov)) * 2.
		      * (np.sqrt(np.pi) * nsigma)**(self.ndim)
		      / (self.ndim * gamma(self.ndim/2.))
		    )
		
		if verbose:
			print 'nsigma = %.3g' % (nsigma / np.sqrt(self.ndim))
			print '<1/p> = %.3g' % sum_inv_p
			print 'V = %.3g' % V
			print 'N = %d' % (np.sum(weight))
		
		return V * np.sum(weight) / sum_inv_p
	
	#
	# Mutators
	#
	
	def set_p_method(self, stretch=0.5):
		'''
		Sets the fraction of times various step types are used. If the
		sum is less than 1.0, then the remainder goes towards
		Metropolis-Hastings steps.
		
		Inputs:
		    stretch  Fraction of steps which use the affine stretch
		             procedure (default: 0.5).
		'''
		self.p_stretch = stretch
	
	def set_affine_scale(self, scale=2.):
		'''
		Sets the scale of affine stretch steps.
		
		Inputs:
		    scale  A value around 2.0 is generally appropriate. Must be
		           greater than 1.0 (default: 2.0).
		'''
		if scale <= 1.0:
			raise ValueError('scale must be greater than 1.0')
		self.a = scale
		self.sqrt_a = np.sqrt(scale)
	
	def update_MH_cov(self, scale=0.1):
		'''
		Set the Metropolis-Hastings proposal covariance to some scaling
		times the covariance of the chain.
		
		Inputs:
		    scale  Amount by which to scale the covariance matrix
		           (default: 0.1).
		'''
		self.cov = scale * self.get_cov()
	
	def clear(self):
		'''
		Clears the history of the chain, the statistics and the
		diagnostics, but retains the current state.
		'''
		# State
		self.weight_X = np.zeros(self.size, dtype='i8')
		
		# Chain
		self.chain = []
		self.weight = []
		self.lnp = []
		
		# Statistics
		self.E_k = np.zeros(self.ndim, dtype='f8')
		self.E_ij = np.zeros((self.ndim, self.ndim), dtype='f8')
		self.sum_weight = 0
		
		# Diagnostics
		self.N_accepted = 0
		self.N_rejected = 0
	
	def standard_run(self, N_steps, MH_scale=0.2, affine_scale=2.0,
	                                                     verbose=False):
		'''
		Runs the burn-in and main phase in a fairly standard manner. The
		burn-in takes up half the number of steps, and is discarded.
		There is no automatic tuning, but the Metropolis-Hastings
		scale may be set.
		
		Inputs:
		    N_steps       # of steps total, including burn-in.
		    
		    MH_scale      Metropolis-Hastings step scale (the MH proposal
		                  covariance is the chain covariance multiplied
		                  by this scale) (default: 0.2).
		    
		    affine_scale  Scale of affine stretch steps (default: 2.0).
		    
		    verbose       Enables verbose output (default: False).
		'''
		print 'Burn-in phase #1: Settling into high-probability region ...'
		for i in xrange(int(N_steps*0.1)):
			self.step()
		
		print 'Burn-in phase #2: Exploratory run to determine covariance ...'
		self.clear()
		for i in xrange(int(N_steps*0.1)):
			self.step()
		
		print 'Burn-in phase #3: Mixing further ...'
		if self.p_stretch < 1.:
			self.update_MH_cov(scale=MH_scale)
		self.set_affine_scale(affine_scale)
		for i in xrange(int(N_steps*0.3)):
			self.step()
		
		print 'Main phase ...'
		if self.p_stretch < 1.:
			self.update_MH_cov(scale=MH_scale)
		self.clear()
		for i in xrange(int(N_steps*0.5)):
			self.step()
		self.flush()


ndim = 20
ln_norm = -(ndim/2.)*np.log(2.*np.pi)

def ln_p(X, *args, **kwargs):
	tmp = X - 3.
	ln_p_tmp = ln_norm -0.5 * np.einsum('ni,ni->n', tmp, tmp)
	tmp = X + 3.
	ln_p_tmp += ln_norm -0.5 * np.einsum('ni,ni->n', tmp, tmp)
	return ln_p_tmp

def rand_state(n):
	return np.random.normal(loc=0., scale=10., size=(n,ndim))

def main():
	cov = np.diag([0.1 for i in xrange(ndim)])
	
	sampler = TMCMC(ln_p, rand_state, cov, size=1000)
	sampler.set_p_method(stretch=1.)
	sampler.standard_run(5000, verbose=True)
	
	print 'Determining statistics ...'
	mean = sampler.get_mean()
	cov = sampler.get_cov()
	
	print 'Acceptance rate: %.3f %%' % (100.*sampler.get_acceptance())
	print ''
	print 'Mean:'
	print mean
	print ''
	print 'Covariance:'
	print cov
	print ''
	print 'Z = %.3g' % (sampler.get_Z_harmonic(use_mean=False, nsigma=2.,
	                                           frac=0.1, verbose=True))
	
	return 0

if __name__ == '__main__':
	main()

