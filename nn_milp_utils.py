#!/usr/bin/python

import argparse
import os
import numpy as np
import time
from gurobipy import *
import re

import random

from common.io import mkpath, mkdir
from dir_lookup import *


################################################################################
# Line Parser
################################################################################
def parse_line(line):
    my_list = re.split(', |,| |\[|\]|\];', line)
    # Ignore empty strings
    # my_list = filter(None, my_list)
    # <THIAGO>
    new_list = []
    for i in my_list:
        if len(i)>0:
            new_list.append(i)
    my_list = new_list
    # </THIAGO>
    output  = np.array(my_list)

    return output


################################################################################
# Returns the actual index in the matrix
################################################################################
def parse_file(input):

    word1 = "levels = "
    word2 = "n = ["
    word3 = "W ="
    word4 = "B ="
    word5 = "];"


    # Output variables
    layers = 0   # layer number at which we have the output
    weights = [] # list whose each elements contains weight matrices
    bias    = [] # list whose each element contain biases

    # IMP NOTE -
    # Bias should be in the form of matrices. eg if we have 2 bias term
    # for a layer, the bias element shape should be (2,1) and not (2, )

    # Assumes that each row of each weight matrix is written in 1 line
    weight_flag = False
    weight_line_cnt  = 0

    # Assumes that bias for 1 layer is written in 1 line
    bias_flag   = True
    bias_layer_cnt = 0

    with open(input, 'r') as fp:
        for cnt, line in enumerate(fp):
            # Remove trailing characters
            line = line.strip()

            # if not an empty line
            if(len(line) > 0):

                # Comment found. Skip
                if (line[0:2] == "//"):
                    if (line[0:3] == "//C"):
                        tokens = line.split()
                        global accuracy
                        accuracy = float(tokens[3].split("%")[0])
                    pass

                elif (word1.lower() in line):
                    layers = int(line[len(word1)])

                elif (word2.lower() in line):
                    temp            = line[len(word2):-2]
                    nodes_per_layer = parse_line(temp).astype(int)
                    line_bins       = np.cumsum(nodes_per_layer[1:])

                elif (word3.lower() in line.lower()):
                    weight_flag = True

                elif (word4.lower() in line.lower()):
                    bias_flag = True

                else:
                    if (weight_flag):
                        if (word5 in line):
                            weight_flag = False
                        else:
                            # These will be weights
                            # Get which weight matrix the current line goes
                            index = np.digitize(weight_line_cnt, line_bins) #np digitize is 1 ordered

                            # Need a new weight matrix
                            if (weight_line_cnt == 0 or weight_line_cnt in line_bins):
                                weights.append(np.zeros((nodes_per_layer[index + 1], nodes_per_layer[index])))
                                row_cnt = 0

                            # row_cnt keeps track of the row of the weight matrix to write this line
                            weights[index][row_cnt, :] = parse_line(line).astype(np.float)

                            row_cnt += 1
                            weight_line_cnt += 1

                    elif (bias_flag):
                        if (word5 in line):
                            bias_flag = False
                        else:
                            # These will be biases
                            temp = parse_line(line).astype(np.float)
                            bias.append(np.transpose([temp]))

    return layers, nodes_per_layer, weights, bias


################################################################################
# Print bounds
################################################################################
def print_bounds(tot_layers, nodes_per_layer, bounds):
    max_nodes = np.max(nodes_per_layer[1:])

    np.set_printoptions(threshold=np.inf, formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("\nMaxima for the nodes")
    print(bounds[1:, 0:max_nodes, 0])
    print("Minima of the nodes")
    print(bounds[1:, 0:max_nodes, 1])

    r,c = np.where(bounds[1:, 0:max_nodes, 0] <= 0)

    print("")
    #print("------------------------------------------------------------------------")
    if (r.shape[0] > 0):
        print("Number_stably_inactive_nodes {}".format(r.shape[0]))
        #print("------------------------------------------------------------------------")
    else:
        print("Number_stably_inactive_nodes {}".format(0))

    with open(os.path.join(os.path.dirname(input), inactive_nodes_file), 'w') as the_file:
        for i in range(r.shape[0]):
            l = r[i] + 1   #+1 since we have ignored the input nodes while printing
            u = c[i]
            #print("(%d, %d)" %(l, u))
            the_file.write(str(l) + " " + str(u) + "\n")

    r,c = np.where(bounds[1:, 0:max_nodes, 1] <= 0)

    #print("------------------------------------------------------------------------")
    if (r.shape[0] > 0):
        print("Number_stably_active_nodes   {}".format(r.shape[0]))
        #print("------------------------------------------------------------------------")
    else:
        print("Number_stably_active_nodes   {}".format(0))


def mycallback(model, where):
    global positive_solution

    if where == GRB.Callback.MIP:
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if objbnd < 0:
            #print(" * NEGATIVE BOUND *")
            model.terminate()
            pass
        if objbnd < GRB.INFINITY and positive_solution:
            #print(" * POSITIVE BOUND WITH POSITIVE SOLUTION *", objbnd)
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj > 0:
            #print(" * (POSITIVE SOLUTION) *")
            positive_solution = True

def layercallback(model, where):
    global p, q, i, nodes_per_layer, positive_units, negative_units
    global h, g

    if where == GRB.Callback.MIPSOL:
        print("FOUND A SOLUTION")
        p_value = model.cbGetSolution(p)
        q_value = model.cbGetSolution(q)
        g_value = model.cbGetSolution(g)
        for n in range(nodes_per_layer[i]):
            if p_value[n] == 1:
                positive_units.add(n)
                model.cbLazy(p[n] == 0)
                #print("+",n,g_value[i,n])
            elif q_value[n] == 1:
                negative_units.add(n)
                model.cbLazy(q[n] == 0)
                #print("-",n,g_value[i,n])
            else:
                pass
                #print("?",n,g_value[i,n])
    elif where == GRB.Callback.MIP:
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        print("BOUND:", objbnd)
        if objbnd<0.5:
            model.terminate()
    elif where == GRB.Callback.MIPNODE:
        print("MIPNODE")
        vars = []
        values = []

        if inject_relaxed_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
            values = model.cbGetNodeRel(vars)
            model.cbSetSolution(vars,values)
        elif inject_random_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
                values.append(bounds[0, input, 0] + random.random()*(bounds[0, input, 1]-bounds[0, input, 0]))
            model.cbSetSolution(vars,values)

        #obj = model.cbUseSolution()
        #print("GOT",obj)


################################################################################
# Custom callback function
# Termination is normally handled through Gurobi parameters
# (MIPGap, NodeLimit, etc.).  You should only use a callback for
# termination if the available parameters don't capture your desired
# termination criterion.
#
# Reference:
# https://www.gurobi.com/documentation/8.1/examples/callback_py.html
################################################################################
def mycallback(model, where):
    # General MIP callback
    if where == GRB.Callback.MIP:
        obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if (obj_bnd < 0):
            print("Objective bound (Soln of relaxed LP) < 0")
            model.terminate()

    # If an MIP solution is found
    elif where == GRB.Callback.MIPSOL:
        model._sol_count += 1
        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        #print("\nObjective f = %f" %(obj_val))

        if (obj_val > 0):
            model._val_sol_count += 1

            if (model._val_sol_count % print_freq == 0):
                sys.stdout.write("Valid/Total Solutions %d / %d Time %d s\n" %(model._val_sol_count, model._sol_count, int(time.time() - now) ))
                sys.stdout.flush()

            #print ("g = ")
            #print (model.cbGetSolution(g))
            #print ("h = ")
            #print (model.cbGetSolution(h))
            #print ("hBar = ")
            #print (model.cbGetSolution(hbar))
            #print ("Binary Variables z = ")
            #print (model.cbGetSolution(z))

            # We want to remove the solution that we just found, so that it does
            # not repeat and also to avoid the solver from finishing before
            # enumerating all positive solutions because it found a solution
            # that provably maximizes f
            #
            # The current solution will have
            # Sum of variables which are 1 - Sum of variables which are 0 = #Variables that are 1
            # So, we add a lazy cut
            # Sum of variables which are 1 - Sum of variables which are 0 <= #Variables that are 1 - 1

            # https://groups.google.com/forum/#!topic/gurobi/d38iycxUIps
            vals = model.cbGetSolution(z)
            expr = LinExpr(0.0)
            ones_cnt = 0

            line_to_write = ""

            # No need for input activations
            for m in range(1, run_till_layer_index):
                for n in range(nodes_per_layer[m]):
                    if (vals[m, n] > 0.9 ):
                        expr.add(model._z[m, n], 1.0)
                        ones_cnt += 1
                        term = "1 "
                    else:
                        expr.add(model._z[m, n], -1.0)
                        term = "0 "

                    line_to_write += term

            line_to_write += "\n"
            # Write the line to the file
            model._my_file.write(line_to_write)
            if (show_activations):
                sys.stdout.write(line_to_write)
                sys.stdout.flush()

            # Add a lazy constraint so that this solution does not appear again
            constraint = model.cbLazy(expr <= ones_cnt-1)
            # print("Ones_cnt = %d" %(ones_cnt))
            # print(expr)
        else:
            pass
            # print("Invalid Solution Found")

