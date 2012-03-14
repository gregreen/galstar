#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       kmeans.py
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


def k_means_dist2(x1, x2):
	Delta = np.matrix(x1 - x2)
	return Delta * Delta.T

def k_means(x_list, means_guess, iterations=5):
	# Get general number of clusters, etc., and set up initial state
	k = len(means_guess)	# Number of clusters to find
	N_dim = len(x_list[0])	# Dimensionality of each point
	M = len(x_list)			# Number of points to be ordered
	mu = np.empty((k, N_dim), dtype=np.float)			# Cluster means
	cluster = np.empty((k, M, N_dim), dtype=np.float)	# Points in each cluster
	length = np.empty(k, dtype=int)						# Number of points in each cluster
	
	# Initialize means
	for i in range(k):
		mu[i] = means_guess[i].__copy__()
	
	for n in range(iterations):
		# Assign each point to the cluster with the closest center
		length.fill(0)
		for i in range(M):
			d2_min = 1e100
			closest = 0
			for j in range(k):
				d2_tmp = k_means_dist2(x_list[i], mu[j])
				if d2_tmp < d2_min:
					d2_min = d2_tmp
					closest = j
			cluster[closest, length[closest]] = x_list[i].__copy__()
			length[closest] += 1
		
		# Set center of each cluster to mean of member points
		for i in range(k):
			mu[i] = cluster[i,:length[i]].mean(0)
	
	# Return the set of clusters in a list
	cluster_list = []
	for i in range(k):
		cluster_list.append(cluster[i,:length[i]])
	return cluster_list


def k_means_affine(x_list, means_guess, iterations=5, return_clusters=False):
	# Get general number of clusters, etc., and set up initial state
	k = len(means_guess)	# Number of clusters to find
	N_dim = len(x_list[0])	# Dimensionality of each point
	M = len(x_list)			# Number of points to be ordered
	mu = np.empty((k, N_dim), dtype=np.float)			# Cluster means
	#cluster = np.empty((k, M, N_dim), dtype=np.float)	# Points in each cluster
	cluster_mask = np.empty((k, M), dtype=bool)			# Mask for each cluster
	
	# Apply an affine transformation which diagonalizes the covariance matrix
	cov = np.cov(np.array(x_list).T)
	eigval, eigvec = np.linalg.eigh(cov)
	eigval = np.sqrt(eigval)
	x_list_unit = np.empty((M, N_dim), dtype=np.float)
	for i in range(M):
		x_list_unit[i] = np.array((eigvec * np.matrix(x_list[i]).T).T)[0]
		for j in range(N_dim):
			x_list_unit[i,j] /= eigval[j];
	
	x_arr = np.array(x_list)
	
	# Initialize means
	for i in range(k):
		tmp_mu = np.array((eigvec * np.matrix(means_guess[i]).T).T)[0]
		for j in range(N_dim):
			tmp_mu[j] /= eigval[j];
		mu[i] = tmp_mu.__copy__()
	
	for n in range(iterations):
		# Assign each point to the cluster with the closest center
		cluster_mask.fill(False)
		for i in range(M):
			d2_min = 1e100
			closest = 0
			for j in range(k):
				d2_tmp = k_means_dist2(x_list_unit[i], mu[j])
				if d2_tmp < d2_min:
					d2_min = d2_tmp
					closest = j
			cluster_mask[closest, i] = True
		
		# Set center of each cluster to mean of member points
		for i in range(k):
			mu[i] = x_arr[cluster_mask[i]].mean(0)
	
	if return_clusters: # Return a mask defining the clusters, and the points in each cluster
		cluster_list = []
		for i in range(k):
			cluster_list.append(x_arr[cluster_mask[i]])
		return cluster_mask, cluster_list
	else:				# Only return a mask defining the clusters
		return cluster_mask


def main():
	
	return 0

if __name__ == '__main__':
	main()

