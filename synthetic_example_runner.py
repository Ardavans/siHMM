


from __future__ import division
__author__ = 'saeedi'


from matplotlib import pyplot as plt
from os.path import join, dirname, isfile
import argparse
from argparse import RawTextHelpFormatter
from pyhsmm.pybasicbayes import distributions
from pyhsmm import models, distributions
from pyhsmm.util.general import sgd_passes, hold_out, get_file
from pyhsmm.util.text import progprint_xrange, progprint
import numpy as np
import scipy.io
import os
import sys
from matplotlib import rc
import pyhsmm
from pyhsmm.util.text import progprint_xrange
import collections
import itertools
import operator
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)






####################################### Generate the data #####################################################
np.random.seed(4) #4
obs_dim = 1
xlim = 800
init_state_concentration = 1000.
alpha = 10
gamma = 1
N = 9 #9
sgd_or_mf = 'sgd' #'mf' or 'sgd' choose between these two
num_seqs = 10 #num of batches
num_samples = 10000
obs_hypparams = {'h_0':np.ones(obs_dim) * 0,
            'J_0':np.ones(obs_dim) * 0.01, #sq_0 #changes the hidden state detection (the lower the better) #0.01
            'alpha_0':np.ones(obs_dim) * 100, #(make the number of hidden states worse higher the better) #2
            'beta_0':np.ones(obs_dim) * 100} #1
obs_distns = [distributions.GaussianNonconjNIG(**obs_hypparams) for state in xrange(N)]
HDPHMMSVImodel = models.HMMSegStickyHDP(kappa = 0, alpha=alpha,gamma=gamma,init_state_concentration=init_state_concentration,
        obs_distns=obs_distns, weight_prior_mean = 0.0, weight_prior_std = 0.1, win_size = 1, use_obs_features = False)
for i in range(N):
    print obs_distns[i]

lablefontsize = 19
plt.subplots(4, sharex=True, figsize =(15,7))
f2 = plt.subplot(4, 1, 2)
tempall = HDPHMMSVImodel.generate(num_samples, keep=False)
true_segments_0_1 = tempall[2].true_segmentation
all_states_seq = tempall[1]
all_states_seq = [x if x >= N else x + N for x in all_states_seq]
true_segmentation_prob = tempall[2].true_segmentation_prob
segment_indices = [i for i,val in enumerate(true_segments_0_1) if val==0]


#Making the batches
all_data = tempall[0]
good_length = num_seqs * int(len(all_data)/num_seqs)
all_data_sh = np.array(all_data[:good_length])
# all_data_sh -= np.mean(all_data_sh)
# all_data_sh /= np.std(all_data_sh)
data = np.reshape(all_data_sh, (num_seqs, int(len(all_data_sh)/num_seqs), obs_dim))
data = list(data)


#Coloring
cmap = plt.cm.get_cmap()
for idx, k in enumerate(all_states_seq):
    all_states_seq[idx] = all_states_seq[idx] if all_states_seq[idx] < N else all_states_seq[idx] - N
unused_states = [idx for idx in range(N) if idx not in all_states_seq]
np.random.seed(1)
colorseq = np.random.RandomState(0).permutation(np.linspace(0,1,N))
colors = dict((idx, v if False else cmap(v)) for idx, v in zip(np.array(range(N)),colorseq))
for state in unused_states:
    colors[state] = cmap(1.)
print 'used states: ', N - len(unused_states)
#Plotting only if we have one dimension
if obs_dim == 1:
    for idx, point in enumerate(all_data_sh):
        plt.plot(idx, point, c=colors[all_states_seq[idx]], markersize=8, marker = '.')
else:
    for idx, point in enumerate(all_data_sh):
        plt.plot(idx, all_states_seq[idx], c=colors[all_states_seq[idx]], markersize=8, marker = '.')

for i in segment_indices:
    print i
    plt.axvline(i, c = 'r')
plt.xlim((0,xlim))
plt.ylabel('True Seg.', fontsize = lablefontsize)
f2.axes.xaxis.set_ticklabels([])

f1 = plt.subplot(4, 1, 1)
if obs_dim == 1:
    for idx, point in enumerate(all_data_sh):
        plt.plot(idx, point, markersize=8, marker = '.', c = 'b')

plt.ylabel('Raw Data', fontsize = lablefontsize)
plt.xlim((0,xlim))
f1.axes.xaxis.set_ticklabels([])


################################ MAKE TRAIN AND TEST DATASETS IN BATCHES ##########################




np.random.seed(100)
print 'loading data...'
alldata = data
allseqs = np.array(data)
#datas, heldout = hold_out(allseqs,0.05)

datas = list(allseqs[:allseqs.shape[0] - 1, :])
heldout = list(allseqs[-1:, :])

training_size = sum(data.shape[0] for data in datas)
print '...done!'



###################################################  RUN SVI or MF OVER THE TRAINING SET     ########################
init_state_concentration = 1000.
#kappa = 0.
#alpha_0 = 10 #(1)
alpha = 100#200
gamma = 100#200
kappa_sticky = 1
win_size = 1
use_obs_features = False
N = 30 #10
infseed = 20

obs_hypparams = {'h_0':np.zeros(obs_dim),
            'J_0':np.ones(obs_dim) * 0.001, #sq_0 #changes the hidden state detection (the lower the better) #0.001
            'alpha_0':np.ones(obs_dim) * 0.1, #(make the number of hidden states worse higher the better)
            'beta_0':np.ones(obs_dim) * 1}
# obs_distns = [distributions.ScalarGaussianNonconjNIG(**obs_hypparams) for state in xrange(N)]
obs_distns = [distributions.GaussianNonconjNIG(**obs_hypparams) for state in xrange(N)]

print 'inference observation'
for i in range(N):
    print obs_distns[i]

print 'feature weights before mean field: ', '\n', HDPHMMSVImodel.feature_weights, '\n'

HDPHMMSVImodel = models.HMMSegStickyHDP(obs_dim = 1, kappa = 0, alpha=alpha,gamma=gamma,init_state_concentration=init_state_concentration,
        obs_distns=obs_distns, bern_or_weight = 'weight', svi_or_gibbs = 'svi',
        weight_prior_mean = 0, weight_prior_std = 0.1, win_size = win_size,  use_obs_features = use_obs_features)



np.random.seed(infseed)
if sgd_or_mf == 'mf':
    print 'feature weights before mean field: ', '\n', HDPHMMSVImodel.feature_weights, '\n'
    for i in range(14):
        HDPHMMSVImodel.add_data(datas[i])
    for i in range(20):
        print HDPHMMSVImodel.meanfield_coordinate_descent_step(0.5)
    print 'feature weights after mean field: ', '\n', HDPHMMSVImodel.feature_weights, '\n'
else:
    scores = []
    sgdseq = sgd_passes(tau=0.8,kappa=0.9,datalist=datas, minibatchsize=4,npasses=30) #4, 3
    for t, (data, rho_t) in progprint(enumerate(sgdseq)):
        HDPHMMSVImodel.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(training_size)  , rho_t)
        score = HDPHMMSVImodel.log_likelihood(heldout)
        # print 'feature weights after mean field: ', HDPHMMSVImodel.feature_weights
        print score
        print ""
        if t % 1 == 0:
            scores.append(score)
    # plt.plot(scores)
    # plt.show()



######################################Plotting the states and segments ###########################

f3 = plt.subplot(4, 1, 3)
all_probpairs = []
all_state_seqs = []
all_inferred_segs = []
all_used_states = []
for seq_num in range(num_seqs - 1):
    print 'seq num: ' + str(seq_num)
    nhs = N

    #Need this to choose between SVI and mean field
    if sgd_or_mf == 'sgd':
        s_num = -1
        HDPHMMSVImodel.add_data(datas[seq_num], generate=False)
        HDPHMMSVImodel.states_list[s_num].meanfieldupdate()
    else:
        s_num = seq_num

    #States sequence
    states_seq = HDPHMMSVImodel.states_list[s_num].expected_states.argmax(1).astype('int32')
    states_seq = [x if x >= N else x + N for x in states_seq]
    all_state_seqs.extend(states_seq)
    #States usage
    canonical_ids = collections.defaultdict(itertools.count().next)

    for state in states_seq:
        canonical_ids[state]
    used_states = map(operator.itemgetter(0), sorted(canonical_ids.items(),key=operator.itemgetter(1)))
    print 'used:', np.array(used_states) - N
    all_used_states.extend(used_states)
    segments_seq = []
    for idx, (i, j) in enumerate(zip(HDPHMMSVImodel.states_list[s_num].all_expected_stats[1], HDPHMMSVImodel.states_list[s_num].data)):
        temp_seg = 1- np.argmax((np.sum(i[:nhs]), np.sum(i[nhs:])))
        all_probpairs.append((np.log(np.sum(i[:nhs])), np.log(np.sum(i[nhs:]))))
        segments_seq.append(temp_seg)

    cmap = plt.cm.get_cmap()
    unused_states = [idx for idx in range(N) if idx not in used_states]
    np.random.seed(1)
    colorseq = np.random.RandomState(0).permutation(np.linspace(0,1,N))
    colors = dict((idx, v if False else cmap(v)) for idx, v in zip(np.array(range(N)) + N,colorseq))
    for state in unused_states:
        colors[state] = cmap(1.)

    temp_data = datas[seq_num]
    min_data_point = np.min(temp_data)
    max_data_point = np.max(temp_data)
    #finding the segments
    segment_indices = [i for i, j in enumerate(segments_seq) if j == 1]


    for idx, point in enumerate(temp_data):
        if obs_dim == 1 :
            plt.plot(idx + seq_num * len(temp_data), point, c=colors[states_seq[idx]], marker = '.', markersize=8)
        else:
            plt.plot(idx + seq_num * len(temp_data), states_seq[idx], c=colors[states_seq[idx]], marker = '.', markersize=8)
    current_i = 0
    for i in segment_indices:
        if i - current_i >= 0:
            print 'inf: ', i + seq_num * len(temp_data)
            plt.axvline(i + seq_num * len(temp_data), color='r') #, linewidth=1
            all_inferred_segs.append(i + seq_num * len(temp_data))
        current_i = i
total_log_prob = 0
for idx, seg in enumerate(true_segments_0_1[:len(all_probpairs)]):
    total_log_prob += all_probpairs[idx][1 - int(seg)]
print 'log_prob: ', total_log_prob
print 'used_states: ', len(np.unique(np.array(all_used_states) - N))
plt.ylabel('Inf. Seg.', fontsize = lablefontsize)

plt.xlim((0,xlim))
plt.subplot(4, 1, 4)
break_prob = [np.exp(i[0]) for i in all_probpairs]
plt.plot(range(len(break_prob)), break_prob, ms = '.')
plt.ylabel('Seg. Prob.', fontsize=lablefontsize)
plt.xlabel('Time', fontsize = lablefontsize)
plt.ylim((0,1))
plt.xlim((0,xlim))
f3.axes.xaxis.set_ticklabels([])
plt.tick_params(axis='x', labelsize=16)
plt.show()



































