# -*- coding: utf-8 -*-

import random
import argparse
import dimod
import sys
import networkx as nx
import numpy as np
from dwave.system import LeapHybridSampler

import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def read_in_args():
    # Parameters specified by user

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="random seed for scenario", type=int)
    parser.add_argument("-x", "--width", help="grid width", default=15, type=int)
    parser.add_argument("-y", "--height", help="grid height", default=15, type=int)
    parser.add_argument("-p", "--poi", help="number of POIs", default=3, type=int)
    parser.add_argument("-c", "--chargers", help="number of existing chargers", default=4, type=int)
    parser.add_argument("-n", "--new-chargers", help="umber of new chargers", default=2, type=int)
    args = parser.parse_args()

    w = args.width
    h = args.height
    num_poi = args.poi
    num_cs = args.chargers
    num_new_cs = args.new_chargers


    if (args.seed):
        random.seed(args.seed)

    if (w < 0) or (h < 0) or (num_poi < 0) or (num_cs < 0) or (num_new_cs < 0):
        sys.exit(0)

    if (num_poi > w*h) or (num_cs + num_new_cs > w*h):
        sys.exit(0)

    return args

# Scenario with all previous specified parameters

def set_up_scenario(w, h, num_poi, num_cs):

    G = nx.grid_2d_graph(w, h)
    nodes = list(G.nodes)

    pois = random.sample(nodes, k=num_poi)

    charging_stations = random.sample(nodes, k=num_cs)

    potential_new_cs_nodes = list(G.nodes() - charging_stations)

    return G, pois, charging_stations, potential_new_cs_nodes

def distance(a, b):
    return (a[0]**2 - 2*a[0]*b[0] + b[0]**2) + (a[1]**2 - 2*a[1]*b[1] + b[1]**2)

# Build Sampler to Model Solution

def bqm_build(potential_new_cs_nodes, num_poi, pois, num_cs, charging_stations, num_new_cs):

    gamma1 = len(potential_new_cs_nodes) * 4
    gamma2 = len(potential_new_cs_nodes) / 3
    gamma3 = len(potential_new_cs_nodes) * 1.7
    gamma4 = len(potential_new_cs_nodes) ** 3

    bqm = dimod.AdjVectorBQM(len(potential_new_cs_nodes), 'BINARY')

  # Min average distance to target points
    if num_poi > 0:
        for i in range(len(potential_new_cs_nodes)):
            cand_loc = potential_new_cs_nodes[i]
            avg_dist = sum(distance(cand_loc, loc) for loc in pois) / num_poi
            bqm.linear[i] += avg_dist * gamma1

    # Max distance to existing chargers
    if num_cs > 0:
        for i in range(len(potential_new_cs_nodes)):
            cand_loc = potential_new_cs_nodes[i]
            avg_dist = -sum(distance(cand_loc, loc)
                            for loc in charging_stations) / num_cs
            bqm.linear[i] += avg_dist * gamma2

    # Max distance to other new charging locations
    if num_new_cs > 1:
        for i in range(len(potential_new_cs_nodes)):
            for j in range(i+1, len(potential_new_cs_nodes)):
                ai = potential_new_cs_nodes[i]
                aj = potential_new_cs_nodes[j]
                dist = -distance(ai, aj)
                bqm.add_interaction(i, j, dist * gamma3)

    bqm.update(dimod.generators.combinations(bqm.variables, num_new_cs, strength=gamma4))

    return bqm

def run(bqm, sampler, potential_new_cs_nodes, **kwargs):
    
    # Solve with sampler to find new EV charging locations 


    sampleset = sampler.sample(bqm,
                               label='Example - EV Charger Placement',
                               **kwargs)

    ss = sampleset.first.sample
    new_charging_nodes = [potential_new_cs_nodes[k] for k, v in ss.items() if v == 1]

    return new_charging_nodes
