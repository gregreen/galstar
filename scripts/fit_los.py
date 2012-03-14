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

def draw_rand_los():
	Delta_Ar = np.random.randn(20)
	Delta_Ar = 0.3 * np.multiply(Delta_Ar, Delta_Ar)
	mu_arr = np.linspace(5., 20., 20)
	
	Ar_plot_arr = np.empty(101, dtype=np.float)
	mu_plot_arr = np.linspace(5., 20., 101)
	for i,mu in enumerate(mu_plot_arr):
		tmp = Ar_interp(mu, mu_arr, Delta_Ar)
		Ar_plot_arr[i] = tmp
		print mu, tmp
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(mu_plot_arr, Ar_plot_arr)
	plt.show()

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


def main():
	#draw_rand_los()
	fn_list = [('/home/greg/projects/galstar/output/90_10/DM_Ar_%d.txt' % i) for i in range(0,100)]
	test_p_los(fn_list)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

