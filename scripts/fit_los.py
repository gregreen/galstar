#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       fit_los.py
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

import sys

import numpy as np
from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt;
import matplotlib as mplib

from AffineSampling import AffineSampler
from galstarutils import *


def Ar_interp(mu, mu_arr, Delta_Ar):
	Ar_tmp = 0.
	for i in range(len(mu_arr)):
		if mu >= mu_arr[i]:
			Ar_tmp += Delta_Ar[i]
		else:
			Ar_tmp += Delta_Ar[i] * (mu - mu_arr[i-1]) / (mu_arr[i] - mu_arr[i-1])
			break
	return Ar_tmp

def p_star(p_interpolator, mu_min, mu_max, N_samples, mu_arr, Delta_Ar):
	Delta_mu = (mu_max - mu_min) / (N_samples - 1.)
	
	# Integrate p(mu, Ar) along contour Ar(mu)
	p = 0.
	mu = mu_min
	i, j = 0, 0
	for n in range(N_samples):
		mu += Delta_mu
		Ar = Ar_interp(mu, mu_arr, Delta_Ar)
		p += p_interpolator((mu, Ar))
	
	return Delta_mu * p

def log_p_los(p_interp_list, mu_min, mu_max, N_samples, mu_arr, Delta_Ar):
	logp = 0.
	for interp in p_interp_list:
		logp += np.log(p_star(interp, mu_min, mu_max, N_samples, mu_arr, Delta_Ar))
	return logp

def load_star(fname):
	img, x, y, p = loadimage(fname)
	coords = np.array([x, y]).T
	interp = NearestNDInterpolator(coords, np.exp(p))
	return interp, img, x, y


def get_los(mu_arr, Delta_Ar, N_samples=100):
	Ar_los = np.empty(N_samples, dtype=np.float)
	mu_los = np.linspace(5., 20., N_samples)
	for i,mu in enumerate(mu_los):
		tmp = Ar_interp(mu, mu_arr, Delta_Ar)
		Ar_los[i] = tmp
	return mu_los, Ar_los

def draw_los(ax, mu_arr=None, Delta_Ar=None):
	if mu_arr == None:
		mu_arr = np.linspace(5., 20., 20)
	N = len(mu_arr)
	if Delta_Ar == None:
		Delta_Ar = np.random.randn(N)
		Delta_Ar = 0.3 * np.multiply(Delta_Ar, Delta_Ar)
	
	Ar_plot_arr = np.empty(20*N, dtype=np.float)
	mu_plot_arr = np.linspace(np.min(mu_arr), np.max(mu_arr), 20*N)
	for i,mu in enumerate(mu_plot_arr):
		tmp = Ar_interp(mu, mu_arr, Delta_Ar)
		Ar_plot_arr[i] = tmp
		print mu, tmp
	
	ax.plot(mu_plot_arr, Ar_plot_arr)

def test_p_star(fname):
	Delta_Ar = np.random.randn(20)
	Delta_Ar = 0.1 * np.multiply(Delta_Ar, Delta_Ar)
	mu_arr = np.linspace(5., 20., 20)
	
	interp, img, x, y = load_star(fname)
	
	p = p_star(interp, np.min(x), np.max(x), 1000, mu_arr, Delta_Ar)
	
	print p
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plotimg(img, x, y, ax, axis_labels=None, xlim=(None,None), ylim=(None,None), params=None)
	mu_los, Ar_los = get_los(mu_arr, Delta_Ar, 1000)
	ax.plot(mu_los, Ar_los)

def test_p_los(fname_list):
	Delta_Ar = np.random.randn(20)
	Delta_Ar = 0.1 * np.multiply(Delta_Ar, Delta_Ar)
	mu_arr = np.linspace(5., 20., 20)
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	interp_list = []
	for fname in fname_list:
		interp, img, x, y = load_star(fname)
		interp_list.append(interp)
	
	img, x, y, p_tmp = load_stacked(fname_list)
	plotimg(img, x, y, ax, axis_labels=None, xlim=(None,None), ylim=(None,None), params=None)
	
	mu_los, Ar_los = get_los(mu_arr, Delta_Ar, 1000)
	ax.plot(mu_los, Ar_los)
	
	logp = log_p_los(interp_list, np.min(x), np.max(x), 1000, mu_arr, Delta_Ar)
	print logp


def logpdf(Delta_Ar, params):
	for DAr in Delta_Ar:
		if DAr < 0.:
			return -9.e99
	p_interp_list, mu_min, mu_max, N_samples, mu_arr = params
	return log_p_los(p_interp_list, mu_min, mu_max, N_samples, mu_arr, Delta_Ar)

def rand_state():
	Delta_Ar = np.random.randn(10)
	Delta_Ar = 0.1 * np.multiply(Delta_Ar, Delta_Ar)
	return Delta_Ar

def sample_los(fname_list):
	# Load the data
	interp_list = []
	mu_min, mu_max = None, None
	for i,fname in enumerate(fname_list):
		interp, img, x, y = load_star(fname)
		interp_list.append(interp)
		if i == 0:
			mu_min, mu_max = np.min(x), np.max(x)
	mu_arr = np.linspace(mu_min, mu_max, 10)
	params = interp_list, mu_min, mu_max, 30, mu_arr
	
	# Set up the sampler
	print 'Setting up sampler...'
	sampler = AffineSampler(rand_state, logpdf, params, a=2., L=20)
	
	# Burn-in
	sys.stdout.write('Burning in')
	for i in range(100):
		sampler.step()
		sys.stdout.write('.'); sys.stdout.flush()
	sys.stdout.write('\n'); sys.stdout.flush()
	sampler.reset_chain()
	
	# Main run
	sys.stdout.write('Running chain')
	for i in range(1000):
		sampler.step()
		sys.stdout.write('.'); sys.stdout.flush()
	sys.stdout.write('\n'); sys.stdout.flush()
	
	# Determine the mean of the chain
	Delta_Ar_mean = sampler.get_mean()
	print 'Mean:', Delta_Ar_mean
	
	# Print the probability of the mean
	logp = log_p_los(interp_list, np.min(x), np.max(x), 100, mu_arr, Delta_Ar_mean)
	print 'log(p) = %.3g' % logp
	
	# Plot the stacked pdfs
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plot_stacked(ax, fname_list)
	
	# Plot the mean fit
	mu_los, Ar_los = get_los(mu_arr, Delta_Ar_mean, 1000)
	ax.plot(mu_los, Ar_los)

def plot_stacked(ax, fname_list, stack_linear=False):
	img, x, y, p_tmp = load_stacked(fname_list, stack_linear)
	plotimg(img, x, y, ax, axis_labels=('\mu','A_r'), xlim=(None,None), ylim=(None,None), params=None)

	
	
'''
	0.0517	+-	0.0278
	0.0613	+-	0.0307
	0.0463	+-	0.0304
	0.0557	+-	0.0264
	0.0581	+-	0.0206
	0.0486	+-	0.0248
	0.0439	+-	0.028
	0.0587	+-	0.0344
	0.0558	+-	0.0258
	0.0555	+-	0.0437
'''

def main():
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	mu_arr = np.linspace(5., 20., 10)
	fname_list = [('/home/greg/projects/galstar/output/90_10/DM_Ar_%d.txt' % i) for i in range(1100)]
	plot_stacked(ax, fname_list, stack_linear=True, normalize=True)
	Delta_Ar = np.array([0.0517, 0.0613, 0.0463, 0.0557, 0.0581, 0.0486, 0.0439, 0.0587, 0.0558, 0.0555])
	draw_los(ax, mu_arr, Delta_Ar)
	
	'''index_list = np.arange(1100)
	np.random.shuffle(index_list)
	fn_list = [('/home/greg/projects/galstar/output/90_10/DM_Ar_%d.txt' % i) for i in index_list[:10]]
	sample_los(fn_list)'''
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

