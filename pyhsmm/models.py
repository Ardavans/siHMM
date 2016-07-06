from __future__ import division
import numpy as np
import itertools
import collections
import operator
import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import cm
from warnings import warn
from scipy.sparse import coo_matrix
import scipy as sp
from numpy import newaxis as na
#import autograd.numpy as npa
#import autograd.scipy as spa
# from autograd import grad
# from autograd.util import quick_grad_check
import time

import pyhsmm
from pyhsmm.basic.abstractions import Model, ModelGibbsSampling, \
        ModelEM, ModelMAPEM, ModelMeanField, ModelMeanFieldSVI
from pyhsmm.internals import hmm_states, hsmm_states, hsmm_inb_states, \
        initial_state, transitions
from pyhsmm.util.general import list_split, window
from pyhsmm.util.profiling import line_profiled

################
#  HMM Mixins  #
################


class _HMMBase(Model):
    _states_class = hmm_states.HMMStatesPython
    _trans_class = transitions.HMMTransitions
    _trans_conc_class = transitions.HMMTransitionsConc
    _init_state_class = initial_state.HMMInitialState

    def __init__(self,
            obs_distns,
            trans_distn=None,
            alpha=None,alpha_a_0=None,alpha_b_0=None,trans_matrix=None,
            init_state_distn=None,init_state_concentration=None,pi_0=None):
        self.obs_distns = obs_distns
        self.states_list = []

        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif not None in (alpha_a_0,alpha_b_0):
            self.trans_distn = self._trans_conc_class(
                    num_states=len(obs_distns),
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    trans_matrix=trans_matrix)
        else:
            self.trans_distn = self._trans_class(
                    num_states=len(obs_distns),alpha=alpha,trans_matrix=trans_matrix)

        if init_state_distn is not None:
            if init_state_distn == 'uniform':
                self.init_state_distn = initial_state.UniformInitialState(model=self)
            else:
                self.init_state_distn = init_state_distn
        else:
            self.init_state_distn = self._init_state_class(
                    model=self,
                    init_state_concentration=init_state_concentration,
                    pi_0=pi_0)

        self._clear_caches()

    def add_data(self,data,stateseq=None,**kwargs):
        self.states_list.append(
                self._states_class(
                    model=self,data=data,
                    stateseq=stateseq,**kwargs))

    def generate(self,T,keep=True):
        s = self._states_class(model=self,T=T,initialize_from_prior=True)
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return data, s.stateseq, s

    def _generate_obs(self,s):
        if s.data is None:
            # generating brand new data sequence
            s.data = [s.obs_distns[state].rvs() for idx, state in enumerate(s.stateseq)]
            print len(s.data)
            # counts = np.bincount(s.stateseq,minlength=self.num_states)
            #
            # # obs = [[o.rvs(count)] for o, count in zip(s.obs_distns,counts)]
            # # # for value in obs[3]:
            # # #     print value
            # # s.data = np.squeeze(np.vstack([[i for i in obs[state]] for state in s.stateseq]))
            #
            # obs = [iter(o.rvs(count)) for o, count in zip(s.obs_distns,counts)]
            # s.data = np.squeeze(np.vstack([obs[state].next() for state in s.stateseq])) #np.squeeze(np.vstack([[i for i in obs[state]] for state in s.stateseq])) #np.squeeze(np.vstack([obs[state].next() for state in s.stateseq]))

        else:
            # filling in missing data
            data = s.data
            nan_idx, = np.where(np.isnan(data).any(1))
            counts = np.bincount(s.stateseq[nan_idx],minlength=self.num_states)
            obs = [iter(o.rvs(count)) for o, count in zip(s.obs_distns,counts)]
            for idx, state in zip(nan_idx, s.stateseq[nan_idx]):
                data[idx] = obs[state].next()

        return s.data

    def log_likelihood(self,data=None,**kwargs):
        if data is not None:
            if isinstance(data,np.ndarray):
                self.add_data(data=data,generate=False,**kwargs)
                return self.states_list.pop().log_likelihood()
            else:
                assert isinstance(data,list)
                loglike = 0.
                for d in data:
                    self.add_data(data=d,generate=False,**kwargs)
                    #self._clear_caches()
                    loglike += self.states_list.pop().log_likelihood()
                return loglike
        else:
            return sum(s.log_likelihood() for s in self.states_list)

    def predict(self,seed_data,timesteps,**kwargs):
        full_data = np.vstack((seed_data,np.nan*np.ones((timesteps,seed_data.shape[1]))))
        self.add_data(full_data,**kwargs)
        s = self.states_list.pop()
        s.resample()  # fills in states
        return self._generate_obs(s), s.stateseq  # fills in nan obs

    def predictive_likelihoods(self,test_data,forecast_horizons,num_procs=None,**kwargs):
        assert all(k > 0 for k in forecast_horizons)
        self.add_data(data=test_data,**kwargs)
        s = self.states_list.pop()
        alphal = s.messages_forwards_log()

        cmaxes = alphal.max(axis=1)
        scaled_alphal = np.exp(alphal - cmaxes[:,None])

        if not num_procs:
            prev_k = 0
            outs = []
            for k in forecast_horizons:
                step = k - prev_k
                cmaxes = cmaxes[:-step]
                scaled_alphal = scaled_alphal[:-step].dot(np.linalg.matrix_power(s.trans_matrix,step))

                future_likelihoods = np.logaddexp.reduce(
                        np.log(scaled_alphal) + cmaxes[:,None] + s.aBl[k:],axis=1)
                past_likelihoods = np.logaddexp.reduce(alphal[:-k],axis=1)
                outs.append(future_likelihoods - past_likelihoods)

                prev_k = k
        else:
            from joblib import Parallel, delayed
            import parallel

            parallel.cmaxes = cmaxes
            parallel.alphal = alphal
            parallel.scaled_alphal = scaled_alphal
            parallel.trans_matrix = s.trans_matrix
            parallel.aBl = s.aBl

            outs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_predictive_likelihoods)(k)
                            for k in forecast_horizons)

        return outs

    @property
    def stateseqs(self):
        return [s.stateseq for s in self.states_list]

    @property
    def stateseqs_norep(self):
        return [s.stateseq_norep for s in self.states_list]

    @property
    def durations(self):
        return [s.durations for s in self.states_list]

    @property
    def datas(self):
        return [s.data for s in self.states_list]

    @property
    def num_states(self):
        return len(self.obs_distns)

    @property
    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) \
                + self.num_states**2 - self.num_states

    @property
    def used_states(self):
        'a list of the used states in the order they appear'
        canonical_ids = collections.defaultitertools(itertools.count().next)
        for s in self.states_list:
            for state in s.stateseq:
                canonical_ids[state]
        return map(operator.itemgetter(0),
                sorted(canonical_ids.items(),key=operator.itemgetter(1)))

    @property
    def state_usages(self):
        if len(self.states_list) > 0:
            state_usages = sum(np.bincount(s.stateseq,minlength=self.num_states)
                    for s in self.states_list)
            return state_usages / state_usages.sum()
        else:
            return np.ones(self.num_states)

    ### predicting

    def heldout_viterbi(self,data,**kwargs):
        self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
        s = self.states_list.pop()
        s.Viterbi()
        return s.stateseq

    def heldout_state_marginals(self,data,**kwargs):
        self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
        s = self.states_list.pop()
        s.E_step()
        return s.expected_states

    def _resample_from_mf(self):
        self.trans_distn._resample_from_mf()
        self.init_state_distn._resample_from_mf()
        for o in self.obs_distns:
            o._resample_from_mf()

    ### caching

    def _clear_caches(self):
        for s in self.states_list:
            s.clear_caches()

    def __getstate__(self):
        self._clear_caches()
        return self.__dict__.copy()

    ### plotting

    _fig_sz = 6

    def make_figure(self,**kwargs):
        sz = self._fig_sz

        if len(self.states_list) <= 2:
            fig = plt.figure(figsize=(sz+len(self.states_list),sz),**kwargs)
        else:
            fig = plt.figure(figsize=(2*sz,sz),**kwargs)

        return fig

    def plot(self,fig=None,plot_slice=slice(None),update=False,draw=True):
        update = update and (fig is not None)
        fig = fig if fig else self.make_figure()
        feature_ax, stateseq_axs = self._get_axes(fig)

        sp1_artists = self.plot_observations(feature_ax,plot_slice=plot_slice,update=update)

        assert len(stateseq_axs) == len(self.states_list)
        sp2_artists = \
            [artist for s,ax,data in zip(self.states_list,stateseq_axs,self.datas)
                for artist in self.plot_stateseq(s,ax,plot_slice,update=update,draw=False)]

        if draw: plt.draw()

        return sp1_artists + sp2_artists

    def _get_axes(self,fig):
        # TODO is attaching these to the figure a good idea? why not save them
        # here and reuse them if we recognize the figure being passed in
        sz = self._fig_sz

        if hasattr(fig,'_feature_ax') and hasattr(fig,'_stateseq_axs'):
            return fig._feature_ax, fig._stateseq_axs
        else:
            if len(self.states_list) <= 2:
                gs = GridSpec(sz+len(self.states_list),1)

                feature_ax = plt.subplot(gs[:sz,:])
                stateseq_axs = [plt.subplot(gs[sz+idx]) for idx in range(len(self.states_list))]
            else:
                gs = GridSpec(1,2)
                sgs = GridSpecFromSubplotSpec(len(self.states_list),1,subplot_spec=gs[1])

                feature_ax = plt.subplot(gs[0])
                stateseq_axs = [plt.subplot(sgs[idx]) for idx in range(len(self.states_list))]

            for ax in stateseq_axs:
                ax.grid('off')

            fig._feature_ax, fig._stateseq_axs = feature_ax, stateseq_axs
            return feature_ax, stateseq_axs

    def plot_observations(self,ax=None,color=None,plot_slice=slice(None),update=False):
        ax = ax if ax else plt.gca()
        state_colors = self._get_colors(color)
        scatter_artists = self._plot_2d_data_scatter(ax,state_colors,plot_slice,update)
        param_artists = self._plot_2d_obs_params(ax,state_colors,update)
        return scatter_artists + param_artists

    def _plot_2d_data_scatter(self,ax=None,state_colors=None,plot_slice=slice(None),update=False):
        # TODO this is a special-case hack. breaks for 1D obs. only looks at
        # first two components of ND obs.
        # should only do this if the obs collection has a 2D_feature method
        ax = ax if ax else plt.gca()
        state_colors = state_colors if state_colors else self._get_colors()

        artists = []
        for s, data in zip(self.states_list,self.datas):
            data = data[plot_slice]
            colorseq = [state_colors[state] for state in s.stateseq[plot_slice]]

            if update and hasattr(s,'_data_scatter'):
                s._data_scatter.set_offsets(data[:,:2])
                s._data_scatter.set_color(colorseq)
            else:
                s._data_scatter = ax.scatter(data[:,0],data[:,1],c=colorseq,s=5)
            artists.append(s._data_scatter)

        return artists

    def _plot_2d_obs_params(self,ax=None,state_colors=None,update=False):
        if not all(hasattr(o,'plot') for o in self.obs_distns):
            return []

        keepaxis = ax is not None
        ax = ax if ax else plt.gca()
        axis = ax.axis()

        state_colors = state_colors if state_colors else self._get_colors()
        usages = self.state_usages

        artists = []
        for state, (o, w) in enumerate(zip(self.obs_distns,usages)):
            artists.extend(
                o.plot(
                    color=state_colors[state], label='%d' % state,
                    alpha=min(0.25,1.-(1.-w)**2)/0.25,
                    ax=ax, update=update,draw=False))

        if keepaxis: ax.axis(axis)

        return artists

    def _get_colors(self,color=None,scalars=False,color_method=None):
        color_method = color_method if color_method else 'usage'
        if color is None:
            cmap = cm.get_cmap()

            if color_method == 'usage':
                freqs = self.state_usages
                used_states = sorted(self.used_states, key=lambda x: freqs[x], reverse=True)
            elif color_method == 'order':
                used_states = self.used_states
            else:
                raise ValueError("color_method must be 'usage' or 'order'")

            unused_states = [idx for idx in range(self.num_states) if idx not in used_states]

            colorseq = np.random.RandomState(0).permutation(np.linspace(0,1,self.num_states))
            colors = dict((idx, v if scalars else cmap(v)) for idx, v in zip(used_states,colorseq))

            for state in unused_states:
                colors[state] = cmap(1.)

            return colors
        elif isinstance(color,dict):
            return color
        else:
            return dict((idx,color) for idx in range(self.num_states))

    def plot_stateseq(self,s,ax=None,plot_slice=slice(None),update=False,draw=True):
        s = self.states_list[s] if isinstance(s,int) else s
        ax = ax if ax else plt.gca()
        state_colors = self._get_colors(scalars=True)

        self._plot_stateseq_pcolor(s,ax,state_colors,plot_slice,update)
        data_values_artist = self._plot_stateseq_data_values(s,ax,state_colors,plot_slice,update)

        if draw: plt.draw()

        return [data_values_artist]

    def _plot_stateseq_pcolor(self,s,ax=None,state_colors=None,
            plot_slice=slice(None),update=False,color_method=None):
        # TODO pcolormesh instead of pcolorfast?
        from pyhsmm.util.general import rle

        s = self.states_list[s] if isinstance(s,int) else s
        ax = ax if ax else plt.gca()
        state_colors = state_colors if state_colors \
                else self._get_colors(scalars=True,color_method=color_method)

        if update and hasattr(s,'_pcolor_im') and s._pcolor_im in ax.images:
            s._pcolor_im.remove()

        data = s.data[plot_slice]
        stateseq = s.stateseq[plot_slice]

        stateseq_norep, durations = rle(stateseq)
        datamin, datamax = data.min(), data.max()

        x, y = np.hstack((0,durations.cumsum())), np.array([datamin,datamax])
        C = np.atleast_2d([state_colors[state] for state in stateseq_norep])

        s._pcolor_im = ax.pcolorfast(x,y,C,vmin=0,vmax=1,alpha=0.3)
        ax.set_ylim((datamin,datamax))
        ax.set_xlim((0,len(stateseq)))
        ax.set_yticks([])

    def _plot_stateseq_data_values(self,s,ax,state_colors,plot_slice,update):
        from matplotlib.collections import LineCollection
        from pyhsmm.util.general import AR_striding, rle

        data = s.data[plot_slice]
        stateseq = s.stateseq[plot_slice]

        colorseq = np.tile(np.array([state_colors[state] for state in stateseq[:-1]]),data.shape[1])

        if update and hasattr(s,'_data_lc'):
            s._data_lc.set_array(colorseq)
        else:
            ts = np.arange(len(stateseq))
            segments = np.vstack(
                [AR_striding(np.hstack((ts[:,None], scalarseq[:,None])),1).reshape(-1,2,2)
                    for scalarseq in data.T])
            lc = s._data_lc = LineCollection(segments)
            lc.set_array(colorseq)
            lc.set_linewidth(0.5)
            ax.add_collection(lc)

        return s._data_lc



class _HMMGibbsSampling(_HMMBase,ModelGibbsSampling):
    @line_profiled
    def resample_model(self,num_procs=0):
        self.resample_parameters()
        self.resample_states(num_procs=num_procs)

    @line_profiled
    def resample_parameters(self):
        self.resample_obs_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()

    def resample_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])
        self._clear_caches()

    @line_profiled
    def resample_trans_distn(self):
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self._clear_caches()

    def resample_init_state_distn(self):
        self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])
        self._clear_caches()

    def resample_states(self,num_procs=0):
        if num_procs == 0:
            for s in self.states_list:
                s.resample()
        else:
            self._joblib_resample_states(self.states_list,num_procs)

    def copy_sample(self):
        new = copy.copy(self)
        new.obs_distns = [o.copy_sample() for o in self.obs_distns]
        new.trans_distn = self.trans_distn.copy_sample()
        new.init_state_distn = self.init_state_distn.copy_sample()
        new.states_list = [s.copy_sample(new) for s in self.states_list]
        return new

    ### joblib parallel stuff here

    def _joblib_resample_states(self,states_list,num_procs):
        from joblib import Parallel, delayed
        import parallel

        # warn('joblib is segfaulting on OS X only, not sure why')

        if len(states_list) > 0:
            joblib_args = list_split(
                    [self._get_joblib_pair(s) for s in states_list],
                    num_procs)

            parallel.model = self
            parallel.args = joblib_args

            raw_stateseqs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_sampled_stateseq)(idx)
                            for idx in range(len(joblib_args)))

            for s, (stateseq, log_likelihood) in zip(
                    [s for grp in list_split(states_list,num_procs) for s in grp],
                    [seq for grp in raw_stateseqs for seq in grp]):
                s.stateseq, s._normalizer = stateseq, log_likelihood

    def _get_joblib_pair(self,states_obj):
        return (states_obj.data,states_obj._kwargs)


class _HMMMeanField(_HMMBase,ModelMeanField):
    def meanfield_coordinate_descent_step(self,num_procs=0):
        self._meanfield_update_sweep(num_procs=num_procs)
        return self._vlb()

    def _meanfield_update_sweep(self,num_procs=0):
        # NOTE: we want to update the states factor last to make the VLB
        # computation efficient, but to update the parameters first we have to
        # ensure everything in states_list has expected statistics computed
        self._meanfield_update_states_list(
            [s for s in self.states_list if not hasattr(s,'expected_states')],
            num_procs)

        self.meanfield_update_parameters()
        self.meanfield_update_states(num_procs)

    def meanfield_update_parameters(self):
        self.meanfield_update_obs_distns()
        self.meanfield_update_trans_distn()
        self.meanfield_update_init_state_distn()

    def meanfield_update_obs_distns(self):
        for state, o in enumerate(self.obs_distns):
            o.meanfieldupdate([s.data for s in self.states_list],
                    [s.expected_states[:,state] for s in self.states_list])

    def meanfield_update_trans_distn(self):
        self.trans_distn.meanfieldupdate(
                [s.expected_transcounts for s in self.states_list])

    def meanfield_update_init_state_distn(self):
        self.init_state_distn.meanfieldupdate(
                [s.expected_states[0] for s in self.states_list])

    def meanfield_update_states(self,num_procs=0):
        self._meanfield_update_states_list(self.states_list,num_procs=num_procs)

    def _meanfield_update_states_list(self,states_list,num_procs=0):
        if num_procs == 0:
            for s in states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(states_list,num_procs)

    def _vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += self.trans_distn.get_vlb()
        vlb += self.init_state_distn.get_vlb()
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb

    ### joblib parallel stuff here

    def _joblib_meanfield_update_states(self,states_list,num_procs):
        if len(states_list) > 0:
            from joblib import Parallel, delayed
            import parallel

            joblib_args = list_split(
                    [self._get_joblib_pair(s) for s in states_list],
                    num_procs)

            parallel.model = self
            parallel.args = joblib_args

            allstats = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_stats)(idx) for idx in range(len(joblib_args)))

            for s, stats in zip(
                    [s for grp in list_split(states_list) for s in grp],
                    [s for grp in allstats for s in grp]):
                s.all_expected_stats = stats

    def _get_joblib_pair(self,states_obj):
        return (states_obj.data,states_obj._kwargs)


class _HMMSVI(_HMMBase,ModelMeanFieldSVI):
    # NOTE: classes with this mixin should also have the _HMMMeanField mixin for
    # joblib/multiprocessing stuff to work
    def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize,num_procs=0,**kwargs):
        ## compute the local mean field step for the minibatch
        mb_states_list = self._get_mb_states_list(minibatch,**kwargs)
        if num_procs == 0:
            for s in mb_states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(mb_states_list,num_procs)

        ## take a global step on the parameters
        self._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)

    def _get_mb_states_list(self,minibatch,**kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        mb_states_list = []
        for mb in minibatch:
            self.add_data(mb,generate=False,**kwargs)
            mb_states_list.append(self.states_list.pop())
        return mb_states_list

    def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
        self._meanfield_sgdstep_obs_distns(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_trans_distn(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_init_state_distn(mb_states_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_obs_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, o in enumerate(self.obs_distns):
            o.meanfield_sgdstep(
                    [s.data for s in mb_states_list],
                    [s.expected_states[:,state] for s in mb_states_list],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.trans_distn.meanfield_sgdstep(
                [s.expected_transcounts for s in mb_states_list],
                minibatchfrac,stepsize)

    def _meanfield_sgdstep_init_state_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.init_state_distn.meanfield_sgdstep(
                [s.expected_states[0] for s in mb_states_list],
                minibatchfrac,stepsize)


class _HMMEM(_HMMBase,ModelEM):
    def EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run EM'
        self._clear_caches()
        self._E_step()
        self._M_step()

    def _E_step(self):
        for s in self.states_list:
            s.E_step()

    def _M_step(self):
        self._M_step_obs_distns()
        self._M_step_init_state_distn()
        self._M_step_trans_distn()

    def _M_step_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data for s in self.states_list],
                    [s.expected_states[:,state] for s in self.states_list])

    def _M_step_init_state_distn(self):
        self.init_state_distn.max_likelihood(
                expected_states_list=[s.expected_states[0] for s in self.states_list])

    def _M_step_trans_distn(self):
        self.trans_distn.max_likelihood(
                expected_transcounts=[s.expected_transcounts for s in self.states_list])

    def BIC(self,data=None):
        '''
        BIC on the passed data. If passed data is None (default), calculates BIC
        on the model's assigned data
        '''
        # NOTE: in principle this method computes the BIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        assert data is None and len(self.states_list) > 0, 'Must have data to get BIC'
        if data is None:
            return -2*sum(self.log_likelihood(s.data).sum() for s in self.states_list) + \
                        self.num_parameters() * np.log(
                                sum(s.data.shape[0] for s in self.states_list))
        else:
            return -2*self.log_likelihood(data) + self.num_parameters() * np.log(data.shape[0])


#
#
#
# class _RobustDPBase(Model):
#     _states_class = hmm_states.HDPStates
#     _trans_class = transitions.DATruncHDP
#     #_trans_conc_class = transitions.HMMTransitionsConc
#
#     def __init__(self,
#             obs_distns,
#             num_docs = None,
#             trans_distn=None,
#             alpha=None,alpha_a_0=None,alpha_b_0=None,trans_matrix=None, gamma=None):
#         self.obs_distns = obs_distns
#         self.states_list = []
#         self.num_states = len(obs_distns)
#         if trans_distn is not None:
#             self.trans_distn = trans_distn
#         else:
#             self.trans_distn = self._trans_class(
#                     num_states=self.num_states,num_docs=1, gamma=gamma, alpha=alpha,trans_matrix=trans_matrix)
#
#         self._clear_caches()
#
#     def _clear_caches(self):
#         for s in self.states_list:
#             s.clear_caches()
#
#     def add_data(self,data,stateseq=None,**kwargs):
#         self.states_list.append(
#                 self._states_class(
#                     model=self,data=data, num_states =self.num_states,
#                     stateseq=stateseq,**kwargs))
#
#     def generate(self, T, keep=True):
#         s = self._states_class(model=self,T=T, initialize_from_prior=True)
#         data = self._generate_obs(s)
#         if keep:
#             self.states_list.append(s)
#         return data, s.stateseq, s
#
#     def _generate_obs(self,s):
#         if s.data is None:
#             # generating brand new data sequence
#             s.data = [s.obs_distns[state].rvs() for idx, state in enumerate(s.stateseq)]
#         else:
#             # filling in missing data
#             data = s.data
#             nan_idx, = np.where(np.isnan(data).any(1))
#             counts = np.bincount(s.stateseq[nan_idx],minlength=self.num_states)
#             obs = [iter(o.rvs(count)) for o, count in zip(s.obs_distns,counts)]
#             for idx, state in zip(nan_idx, s.stateseq[nan_idx]):
#                 data[idx] = obs[state].next()
#
#         return s.data
#
#     def log_likelihood(self,data=None, **kwargs):
#         if data is not None:
#             if isinstance(data,np.ndarray):
#                 self.add_data(data=data, generate=False,**kwargs)
#                 return self.states_list.pop().log_likelihood()
#             else:
#                 assert isinstance(data,list)
#                 loglike = 0.
#                 for idx, d in enumerate(data):
#                     self.add_data(data=d, generate=False,**kwargs)
#                     #self._clear_caches()
#                     loglike += self.states_list.pop().log_likelihood()
#                 return loglike
#         else:
#             return sum(s.log_likelihood() for s in self.states_list)
#
#     @property
#     def stateseqs(self):
#         return [s.stateseq for s in self.states_list]
#
#
#
#
# class _RobustDPMeanField(_RobustDPBase,ModelMeanField):
#     def meanfield_coordinate_descent_step(self,num_procs=0):
#         self._meanfield_update_sweep(num_procs=num_procs)
#
#     def _meanfield_update_sweep(self,num_procs=0):
#         # NOTE: we want to update the states factor last to make the VLB
#         # computation efficient, but to update the parameters first we have to
#         # ensure everything in states_list has expected statistics computed
#         self._meanfield_update_states_list(
#             [s for s in self.states_list if not hasattr(s,'expected_states')],
#             num_procs)
#
#         self.meanfield_update_parameters()
#         self.meanfield_update_states(num_procs)
#
#     def meanfield_update_parameters(self):
#         self.meanfield_update_obs_distns()
#         self.meanfield_update_trans_distn()
#         self.meanfield_update_weights()
#
#     def meanfield_update_obs_distns(self):
#         for state, o in enumerate(self.obs_distns):
#             o.meanfieldupdate([np.array([i[0] for i in s.data]) for s in self.states_list],
#                     [[s.expected_states[:,state] * np.array([i[1] for i in s.data])][0] for s in self.states_list])
#
#
#     def meanfield_update_trans_distn(self):
#         self.trans_distn.meanfieldupdate(
#                 [([s.expected_states * np.array([i[1] for i in s.data])[:, na]][0], s.doc_num) for s in self.states_list])
#
#
#     def meanfield_update_states(self,num_procs=0):
#         self._meanfield_update_states_list(self.states_list,num_procs=num_procs)
#
#
#     def meanfiled_update_weights(self):
#         pass
#
#     def _meanfield_update_states_list(self,states_list,num_procs=0):
#         if num_procs == 0:
#             for s in states_list:
#                 s.meanfieldupdate()
#
#
#
# class _HDPSVI(_RobustDPBase,ModelMeanFieldSVI):
#     # NOTE: classes with this mixin should also have the _HMMMeanField mixin for
#     # joblib/multiprocessing stuff to work
#     def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize,num_procs=0,**kwargs):
#         ## compute the local mean field step for the minibatch
#         mb_states_list = self._get_mb_states_list(minibatch,**kwargs)
#         if num_procs == 0:
#             for s in mb_states_list:
#                 s.meanfieldupdate()
#         else:
#             self._joblib_meanfield_update_states(mb_states_list,num_procs)
#
#         ## take a global step on the parameters
#         self._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)
#         print ""
#     def _get_mb_states_list(self,minibatch,**kwargs):
#         minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
#         mb_states_list = []
#         for mb in minibatch: #minibatch is a pair of word sequence and doc index
#             self.add_data(mb, generate=False,**kwargs)
#             mb_states_list.append(self.states_list.pop())
#         return mb_states_list
#
#     def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
#         self._meanfield_sgdstep_obs_distns(mb_states_list,minibatchfrac,stepsize)
#         self._meanfield_sgdstep_trans_distn(mb_states_list,minibatchfrac,stepsize)
#         self._meanfield_sgdstep_weights(mb_states_list,minibatchfrac,stepsize)
#
#     def _meanfield_sgdstep_obs_distns(self,mb_states_list,minibatchfrac,stepsize):
#         for state, o in enumerate(self.obs_distns):
#             o.meanfield_sgdstep(
#                     [np.array([i[0] for i in s.data]) for s in mb_states_list],
#                     [[s.expected_states[:,state] * np.array([i[1] for i in s.data])][0] for s in mb_states_list],
#                     minibatchfrac,stepsize)
#
#     def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
#         self.trans_distn.meanfield_sgdstep(
#                 [([s.expected_states * np.array([i[1] for i in s.data])[:, na]][0], s.doc_num) for s in mb_states_list],
#                 minibatchfrac,stepsize)
#
#     def _meanfield_sgdstep_weights(self,mb_states_list,minibatchfrac,stepsize):
#         pass
#





#The model that we are using for nonparametric segmentation
class _HMMSeg(_HMMBase):
    _states_class = hmm_states.HMMSegExStatesEigen
    _trans_class = transitions.HMMTransitions
    _trans_conc_class = transitions.HMMTransitionsConc
    _init_state_class = initial_state.HMMInitialState


    def __init__(self,
            obs_distns,
            trans_distn=None,
            alpha=None,alpha_a_0=None,alpha_b_0=None,trans_matrix=None,obs_dim = None,
            init_state_distn = None,init_state_concentration=None,pi_0=None, bern_or_weight = 'bern', svi_or_gibbs = 'gibbs', bern_hypers_alpha = 1,
            bern_hypers_beta = 1, segmentation_distns = None, feature_weights = None, win_size = 1,
                 weight_prior_mean = None, weight_prior_std = None, sgd_steps = None, sgd_step_size = None, use_obs_features = True):
        assert bern_or_weight == 'bern' or bern_or_weight == 'weight'
        if bern_or_weight == 'bern': #make sure we have hyperparameters for the Beta prior over Bernoulli
            assert bern_hypers_alpha is not None and bern_hypers_beta is not None
        self.win_size = win_size #size of the overlapping windows
        self.obs_distns = obs_distns
        self.states_list = []
        self.mega_list_y_z_1 = [] #this one and the next are used for the weight optimization part
        self.mega_list_window_data = []
        self.window_list = [] #this is a list of lists; we convert each element of self.states_list (a sequence) to a list
        self.sgd_steps = sgd_steps
        self.sgd_step_size = sgd_step_size
        self.use_obs_features = use_obs_features
        self.bern_or_weight = bern_or_weight
        self.bern_hypers_alpha = bern_hypers_alpha
        self.bern_hypers_beta = bern_hypers_beta
        self.svi_or_gibbs = svi_or_gibbs
        self.obs_dim = obs_dim
        if feature_weights is not None:
            self.feature_weights = feature_weights
        else: #initialize from prior
            num_hidden_states = len(obs_distns) # number of features corresponding to the hidden state z_t
            try :
                number_of_obs_features = win_size * (self.obs_distns[0].K + win_size - 1 if
                                                     isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical)
                                                     else len(self.obs_distns[0].h_0)) #mu_0 or h_0
            except:
                number_of_obs_features = win_size * (self.obs_distns[0].K + win_size - 1 if
                                                     isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical)
                                                     else obs_dim)
            features_size = number_of_obs_features + num_hidden_states + 1 #last element is for the bias term
            self.feature_weights = weight_prior_std * np.random.randn(features_size) + weight_prior_mean #np.zeros(features_size)#
            if not self.use_obs_features:
                self.feature_weights[:-num_hidden_states-1] = 0
            #self.feature_weights[:len(self.obs_distns[0].h_0)] = 0

        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif not None in (alpha_a_0,alpha_b_0):
            self.trans_distn = self._trans_conc_class(
                    num_states=len(obs_distns),
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    trans_matrix=trans_matrix)
        else:
            self.trans_distn = self._trans_class(
                    num_states=len(obs_distns),alpha=alpha,trans_matrix=trans_matrix)

        if init_state_distn is not None:
            if init_state_distn == 'uniform':
                self.init_state_distn = initial_state.UniformInitialState(model=self)
            else:
                self.init_state_distn = init_state_distn
        else:
            self.init_state_distn = self._init_state_class(
                    model=self,
                    init_state_concentration=init_state_concentration,
                    pi_0=pi_0)

        if bern_or_weight == 'bern' and segmentation_distns is not None:
            self.segmentation_distns = segmentation_distns
        elif bern_or_weight == 'bern' and segmentation_distns is None:
            num_hidden_states = len(obs_distns)
            obs_hypparams = dict(alphav_0 = np.array([self.bern_hypers_alpha, self.bern_hypers_beta]))
            self.segmentation_distns = [pyhsmm.distributions.Categorical(**obs_hypparams) for i in xrange(num_hidden_states)]

        self._clear_caches()

    def convert_window_data(self, data_window):
        #convert the categorical variables to indicator variables and keep the gaussians the same as before
        if isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical):
            num_rows = len(data_window)
            num_obs = self.obs_distns[0].K + num_rows - 1 #one for the special initial state which is the Kth + 1 state

            num_cols = num_obs
            converted_data = np.ravel(coo_matrix((np.ones(num_rows), (range(num_rows), data_window)),
                                                 shape=(num_rows, num_cols)).toarray())
        else:
            converted_data = np.ravel(data_window) #if the gaussian is 2d or more dimensional we need one feature for eahc dimension
        return converted_data

    def add_data(self,data,stateseq=None,**kwargs):
        #create the overlapping window data
        if self.win_size >= 2:
            if isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical):
                padded_data = np.insert(data, 0, self.obs_distns[0].K) #append the (max index + 1) as the initial observation
            else:
                try:
                    if len(self.obs_distns[0].h_0) == 0: #this condition is not used
                        padded_data = np.insert(data, 0, np.array([0] * len(self.obs_distns[0].h_0) * (self.win_size - 1)), axis = 0)
                    else:
                        padded_data = np.vstack((np.tile(np.array([0] * len(self.obs_distns[0].h_0)),(self.win_size - 1, 1)), data))
                except:
                    padded_data = np.vstack((np.tile(np.array([0] * self.obs_distns[0].D),(self.win_size - 1, 1)), data))
                    # padded_data = np.insert(data, 0, np.array([0] * self.obs_distns[0].D * (self.win_size - 1)), axis = 0)
        else:
            padded_data = data

        windowed_data = window(padded_data, self.win_size)

        temp_list = []
        for win in windowed_data:
            converted_data = self.convert_window_data(win)
            temp_list.append(converted_data)

        self.states_list.append(
                self._states_class(window_data=temp_list,
                    model=self,data=data,
                    stateseq=stateseq,**kwargs))


class _HMMSegGibbsSampling(_HMMSeg,ModelGibbsSampling):
    def resample_model(self,num_procs=0):
        self.resample_parameters()
        self.resample_states(num_procs=num_procs)



    @line_profiled
    def resample_parameters(self):
        segments_indices = [[i for i, x in enumerate(s.true_segmentation) if x == 0 or i == 0 or i == len(s.true_segmentation)]
                            for s in self.states_list]
        segment_pairs = [[(j + 1, index_list[idx + 1] + 1) for idx, j in enumerate(index_list[:-1])] for index_list in segments_indices]
        segmented_states_list = [s.stateseq[i:j] for idxx, s in enumerate(self.states_list) for i, j in segment_pairs[idxx]]
        segmented_states_list_for_trans = [s for id, s in enumerate(segmented_states_list)
                                                 if len(segmented_states_list[id]) < 2] #removing the sequences with length 1
        #segmented_states_list = [se in seg for seg in segmented_states_list]

        self.segmented_states_list = segmented_states_list
        self.segmented_states_list_for_trans = segmented_states_list_for_trans
        self.resample_obs_distns()
        #print ' print np.random.normal()', np.random.normal()
        self.resample_trans_distn()
        self.resample_init_state_distn()
        self.resample_segmentation_distns()

    def resample_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == (state or state + self.num_states)] for s in self.states_list])
        self._clear_caches()

    @line_profiled
    def resample_trans_distn(self):
        self.trans_distn.resample([np.array([k if k < self.num_states else k - self.num_states for k in s], dtype=int).astype('int32') for s in self.segmented_states_list_for_trans])
        self._clear_caches()

    def resample_init_state_distn(self):
        nhs = self.num_states
        self.init_state_distn.resample([s[0] if s[0] < nhs else s[0] - nhs for s in self.segmented_states_list])
        self._clear_caches()

    def resample_segmentation_distns(self):
        for state, distn in enumerate(self.segmentation_distns):
            distn.resample([s.true_segmentation[s.stateseq == state] for s in self.states_list])
            self._clear_caches()

    def resample_states(self,num_procs=0):
        if num_procs == 0:
            for s in self.states_list:
                s.resample()
        else:
            self._joblib_resample_states(self.states_list,num_procs)

    def copy_sample(self):
        new = copy.copy(self)
        new.obs_distns = [o.copy_sample() for o in self.obs_distns]
        new.trans_distn = self.trans_distn.copy_sample()
        new.init_state_distn = self.init_state_distn.copy_sample()
        new.segmentation_distns = self.segmentation_distns.copy_sample()
        new.states_list = [s.copy_sample(new) for s in self.states_list]
        return new


class _HMMSegSVI(_HMMSeg, ModelMeanFieldSVI):

    # NOTE: classes with this mixin should also have the _HMMMeanField mixin for
    # joblib/multiprocessing stuff to work
    def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize,num_procs=0,**kwargs):
        ## compute the local mean field step for the minibatch
        mb_states_list = self._get_mb_states_list(minibatch,**kwargs)
        if num_procs == 0:
            for s in mb_states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(mb_states_list,num_procs)

        ## take a global step on the parameters
        self._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)


    def _get_mb_states_list(self,minibatch,**kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        mb_states_list = []
        for mb in minibatch:
            self.add_data(mb,generate=False,**kwargs)
            mb_states_list.append(self.states_list.pop())
        return mb_states_list

    def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
        self._meanfield_sgdstep_obs_distns(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_trans_distn(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_init_state_distn(mb_states_list,minibatchfrac,stepsize)
        if self.bern_or_weight == 'weight':
            self._meanfield_sgdstep_weights(mb_states_list,minibatchfrac,stepsize)
        elif self.bern_or_weight == 'bern':
            self._meanfield_sgdstep_bern_state_distn(mb_states_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_obs_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, o in enumerate(self.obs_distns):
            o.meanfield_sgdstep(
                    [s.data for s in mb_states_list],
                    [s.expected_states[:,state] + s.expected_states[:,state + self.num_states] for s in mb_states_list],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.trans_distn.meanfield_sgdstep(
                [s.expected_transcounts_segmentation for s in mb_states_list],
                minibatchfrac,stepsize)

    def _meanfield_sgdstep_init_state_distn(self,mb_states_list,minibatchfrac,stepsize):
        self.init_state_distn.meanfield_sgdstep(
                [s.expected_initial for s in mb_states_list],
                minibatchfrac,stepsize)

    def _meanfield_sgdstep_bern_state_distn(self, mb_states_list, minibatchfrac, stepsize):
        nhs = self.num_states
        for state, o in enumerate(self.segmentation_distns):
            o.meanfield_sgdstep([s for s in mb_states_list], #this line should be a placeholder
                [np.vstack((s.expected_states[:, state], s.expected_states[:, nhs + state])).T
                 for s in mb_states_list],
                minibatchfrac,stepsize)


    def _meanfield_sgdstep_weights(self, mb_states_list,minibatchfrac,stepsize):
        self.feature_weights_meandfield_sgdstep(self.feature_weights, mb_states_list,
                                                minibatchfrac,stepsize)


    def _vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += self.trans_distn.get_vlb()
        vlb += self.init_state_distn.get_vlb()
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb

    def feature_weights_meandfield_sgdstep(self, feature_weights_input, mb_states_list,
                                           minibatchfrac,stepsize):
        nhs = self.num_states
        # print self.feature_weights
        mega_mega_list_y = np.repeat(np.vstack(np.array([s.window_data for s in mb_states_list])), nhs, axis = 0)
        mega_list_window_data = np.vstack(np.array([s.window_data for s in mb_states_list]))
        mega_list_z = np.vstack(np.tile(np.diag(np.ones(nhs)),(len(s.data), 1)) for s in mb_states_list)
        mega_list_y_z = np.hstack((mega_mega_list_y, mega_list_z))
        mega_list_y_z_1 = np.hstack((mega_list_y_z, np.ones((mega_list_y_z.shape[0],1))))



        temp_list = list(np.sum(feature_weights_input[:-nhs-1] * mega_list_window_data, axis = 1))
        temp_mega_mat_num = np.repeat(temp_list, nhs) # repeat every w*y for the hidden states: [y_t = 1, z_t = 0], [y_t= 1, z_t = 1], [y_t = 1, z_t = 2],...
        temp_w_num =  np.array(list(feature_weights_input[-self.num_states-1:-1]) * mega_list_window_data.shape[0])
        temp_num = temp_mega_mat_num + temp_w_num + feature_weights_input[-1] * np.ones(temp_w_num.shape[0]) # adding the bias term to all
        #log of denom
        temp_denom = np.array([sp.misc.logsumexp(np.array([0, i])) for i in np.array(temp_num)])
        temp_mid_result = np.exp(temp_num - temp_denom) #temp_num



        mega_list_expected_states_s_t_1 = np.vstack(np.array([s.all_expected_stats[1][:,nhs:] for s in mb_states_list])) #only the second half of columns which corresponds to s_t = 1
        mega_list_expected_states_s_t_0 = np.vstack(np.array([s.all_expected_stats[1][:,:nhs] for s in mb_states_list]))
        temp_num_rows = mega_list_expected_states_s_t_1.shape[0]
        mega_list_expected_states_s_t_1 = np.reshape(mega_list_expected_states_s_t_1, (nhs * temp_num_rows, 1))
        mega_list_expected_states_s_t_0 = np.reshape(mega_list_expected_states_s_t_0, (nhs * temp_num_rows, 1))
        mega_list_expected_states_s_t_sum = mega_list_expected_states_s_t_0 + mega_list_expected_states_s_t_1 #


        part_1 = np.sum(mega_list_expected_states_s_t_1 * mega_list_y_z_1, axis = 0)
        mid_w = (mega_list_expected_states_s_t_sum * temp_mid_result[:, None]) * mega_list_y_z_1
        part_2 = np.sum(mid_w, axis = 0)
        grad = (part_1 - part_2)
        # print 'grad norm: ', np.linalg.norm(grad, 2)

        try:
            number_of_obs_features = self.win_size * (self.obs_distns[0].K + self.win_size - 1 if
                                                 isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical)
                                                 else len(self.obs_distns[0].mu_0)) #mu_0 or h_0
        except:
            number_of_obs_features = self.win_size * (self.obs_distns[0].K + self.win_size - 1 if
                                                 isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical)
                                                 else self.obs_dim)

        self.feature_weights =  (1-0.001*stepsize) * self.feature_weights + 0.001*stepsize  * grad #** 1./minibatchfrac
        if not self.use_obs_features:
            self.feature_weights[:number_of_obs_features] = 0


class _HMMSegMeanfield(_HMMSeg, ModelMeanField):

    def meanfield_coordinate_descent_step(self,stepsize, num_procs=0):
        self._meanfield_update_sweep(stepsize, num_procs=num_procs)
        return self._vlb()

    def _meanfield_update_sweep(self,stepsize, num_procs=0):
        # NOTE: we want to update the states factor last to make the VLB
        # computation efficient, but to update the parameters first we have to
        # ensure everything in states_list has expected statistics computed
        self._meanfield_update_states_list(
            [s for s in self.states_list if not hasattr(s,'expected_states')],
            num_procs)

        self.meanfield_update_parameters(stepsize)
        self.meanfield_update_states(num_procs)

    def meanfield_update_parameters(self, stepsize):
        self.meanfield_update_obs_distns()
        self.meanfield_update_trans_distn()
        self.meanfield_update_init_state_distn()
        #self.meanfield_update_weights(stepsize)
        if self.bern_or_weight == 'weight':
            self.meanfield_update_weights(stepsize)
        elif self.bern_or_weight == 'bern':
            self.meanfield_update_bern_state_distn()

    def meanfield_update_obs_distns(self):
        for state, o in enumerate(self.obs_distns):
            o.meanfieldupdate(
                    [s.data for s in self.states_list],
                    [s.expected_states[:,state] + s.expected_states[:,state + self.num_states] for s in self.states_list])


    def meanfield_update_trans_distn(self):
        self.trans_distn.meanfieldupdate(
                [s.expected_transcounts_segmentation for s in self.states_list])

    def meanfield_update_init_state_distn(self):
        self.init_state_distn.meanfieldupdate(
                [s.expected_initial for s in self.states_list])

    def meanfield_update_weights(self,stepsize):
        self.feature_weights_meandfield_update(self.feature_weights,stepsize)

    def meanfield_update_bern_state_distn(self):
        nhs = self.num_states
        for state, o in enumerate(self.segmentation_distns):
            o.meanfield_sgdstep([s.stateseq for s in self.states_list], #this line should be a placeholder
                [np.hstack((s.expected_states[:, state], s.expected_states[:, nhs + state]))
                 for s in self.states_list])

    def meanfield_update_states(self,num_procs=0):
        self._meanfield_update_states_list(self.states_list,num_procs=num_procs)

    def _meanfield_update_states_list(self,states_list,num_procs=0):
        if num_procs == 0:
            for s in states_list:
                s.meanfieldupdate()
        else:
            self._joblib_meanfield_update_states(states_list,num_procs)

    def _vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += self.trans_distn.get_vlb()
        vlb += self.init_state_distn.get_vlb()
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb

    def feature_weights_meandfield_update(self, feature_weights_input, stepsize):
        nhs = self.num_states

        mega_mega_list_y = np.repeat(np.vstack(np.array([s.window_data for s in self.states_list])), nhs, axis = 0)
        mega_list_window_data = np.vstack(np.array([s.window_data for s in self.states_list]))
        mega_list_z = np.vstack(np.tile(np.diag(np.ones(nhs)),(len(s.data), 1)) for s in self.states_list)
        mega_list_y_z = np.hstack((mega_mega_list_y, mega_list_z))
        mega_list_y_z_1 = np.hstack((mega_list_y_z, np.ones((mega_list_y_z.shape[0],1))))



        temp_list = list(np.sum(feature_weights_input[:-nhs-1] * mega_list_window_data, axis = 1))
        temp_mega_mat_num = np.repeat(temp_list, nhs) # repeat every w*y for the hidden states: [y_t = 1, z_t = 0], [y_t= 1, z_t = 1], [y_t = 1, z_t = 2],...
        temp_w_num =  np.array(list(feature_weights_input[-self.num_states-1:-1]) * mega_list_window_data.shape[0])
        temp_num = temp_mega_mat_num + temp_w_num + feature_weights_input[-1] * np.ones(temp_w_num.shape[0]) # adding the bias term to all
        #log of denom
        temp_denom = np.array([sp.misc.logsumexp(np.array([0, i])) for i in np.array(temp_num)])
        temp_mid_result = np.exp(temp_num - temp_denom) #temp_num



        mega_list_expected_states_s_t_1 = np.vstack(np.array([s.all_expected_stats[1][:,nhs:] for s in self.states_list])) #only the second half of columns which corresponds to s_t = 1
        mega_list_expected_states_s_t_0 = np.vstack(np.array([s.all_expected_stats[1][:,:nhs] for s in self.states_list]))
        temp_num_rows = mega_list_expected_states_s_t_1.shape[0]
        mega_list_expected_states_s_t_1 = np.reshape(mega_list_expected_states_s_t_1, (nhs * temp_num_rows, 1))
        mega_list_expected_states_s_t_0 = np.reshape(mega_list_expected_states_s_t_0, (nhs * temp_num_rows, 1))
        mega_list_expected_states_s_t_sum = mega_list_expected_states_s_t_0 + mega_list_expected_states_s_t_1 #


        part_1 = np.sum(mega_list_expected_states_s_t_1 * mega_list_y_z_1, axis = 0)
        mid_w = (mega_list_expected_states_s_t_sum * temp_mid_result[:, None]) * mega_list_y_z_1
        part_2 = np.sum(mid_w, axis = 0)
        grad = (part_1 - part_2)
        #print grad
        print 'grad norm: ', np.linalg.norm(grad, 2)

        number_of_obs_features = self.win_size * (self.obs_distns[0].K + self.win_size - 1 if
                                                 isinstance(self.obs_distns[0], pyhsmm.distributions.Categorical)
                                                 else len(self.obs_distns[0].h_0)) #mu_0 or h_0

        self.feature_weights =  (1-0.001*stepsize) * self.feature_weights + 0.001*stepsize  * grad #** 1./minibatchfrac
        if not self.use_obs_features:
            self.feature_weights[:number_of_obs_features] = 0




# class _HMMSegGibbsSampling(_HMMSeg, ModelGibbsSampling):
#     #this doesn't work as we don't have gibbs for the transition matrix
#     def resample_model(self,num_procs=0):
#         self.resample_parameters()
#         self.resample_states(num_procs=num_procs)
#
#     def resample_parameters(self):
#         self.resample_obs_distns()
#         self.resample_trans_distn()
#         self.resample_init_state_distn()
#
#     def resample_obs_distns(self):
#         for state, distn in enumerate(self.obs_distns):
#             distn.resample([s.data[s.stateseq == state] for s in self.states_list])
#         self._clear_caches()
#
#     def resample_trans_distn(self):
#         self.trans_distn.resample([s.stateseq for s in self.states_list])
#         self._clear_caches()
#
#     def resample_init_state_distn(self):
#         self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])
#         self._clear_caches()
#
#     def resample_states(self,num_procs=0):
#         if num_procs == 0:
#             for s in self.states_list:
#                 s.resample()
#         else:
#             self._joblib_resample_states(self.states_list,num_procs)
#
#     def copy_sample(self):
#         new = copy.copy(self)
#         new.obs_distns = [o.copy_sample() for o in self.obs_distns]
#         new.trans_distn = self.trans_distn.copy_sample()
#         new.init_state_distn = self.init_state_distn.copy_sample()
#         new.states_list = [s.copy_sample(new) for s in self.states_list]
#         return new




class _HMMViterbiEM(_HMMBase,ModelMAPEM):
    def Viterbi_EM_fit(self, tol=0.1, maxiter=20):
        return self.MAP_EM_fit(tol, maxiter)

    def Viterbi_EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run Viterbi EM'
        self._clear_caches()
        self._Viterbi_E_step()
        self._Viterbi_M_step()

    def _Viterbi_E_step(self):
        for s in self.states_list:
            s.Viterbi()

    def _Viterbi_M_step(self):
        self._Viterbi_M_step_obs_distns()
        self._Viterbi_M_step_init_state_distn()
        self._Viterbi_M_step_trans_distn()

    def _Viterbi_M_step_obs_distns(self):
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data[s.stateseq == state] for s in self.states_list])

    def _Viterbi_M_step_init_state_distn(self):
        self.init_state_distn.max_likelihood(
                samples=np.array([s.stateseq[0] for s in self.states_list]))

    def _Viterbi_M_step_trans_distn(self):
        self.trans_distn.max_likelihood([s.stateseq for s in self.states_list])

    MAP_EM_step = Viterbi_EM_step  # for the ModelMAPEM interface


class _WeakLimitHDPMixin(object):
    def __init__(self,
            obs_distns,
            trans_distn=None,alpha=None,alpha_a_0=None,alpha_b_0=None,
            gamma=None,gamma_a_0=None,gamma_b_0=None,trans_matrix=None,kappa = None,
            **kwargs):

        if trans_distn is not None:
            trans_distn = trans_distn
        elif not None in (alpha_a_0,alpha_b_0):
            trans_distn = self._trans_conc_class(
                    num_states=len(obs_distns),
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0,
                    trans_matrix=trans_matrix)
        elif kappa is not None:
            trans_distn = self._trans_class(
                    num_states=len(obs_distns),alpha=alpha,gamma=gamma,
                    trans_matrix=trans_matrix, kappa=kappa)
        else:
            trans_distn = self._trans_class(
                    num_states=len(obs_distns),alpha=alpha,gamma=gamma,
                    trans_matrix=trans_matrix)

        super(_WeakLimitHDPMixin,self).__init__(
                obs_distns=obs_distns, trans_distn=trans_distn, **kwargs)


class _HMMPossibleChangepointsMixin(object):
    _states_class = hmm_states.HMMStatesPossibleChangepoints

    def add_data(self,data,changepoints=None,**kwargs):
        super(_HMMPossibleChangepointsMixin,self).add_data(
                data=data,changepoints=changepoints,**kwargs)

    def _get_mb_states_list(self,minibatch,changepoints=None,**kwargs):
        if changepoints is not None:
            if not isinstance(minibatch,(list,tuple)):
                assert isinstance(minibatch,np.ndarray)
                assert isinstance(changepoints,list) and isinstance(changepoints[0],tuple)
                minibatch = [minibatch]
                changepoints = [changepoints]
            else:
                assert isinstance(changepoints,(list,tuple))  \
                        and isinstance(changepoints[0],(list,tuple)) \
                        and isinstance(changepoints[0][0],tuple)
                assert len(minibatch) == len(changepoints)

        changepoints = changepoints if changepoints is not None \
                else [None]*len(minibatch)

        mb_states_list = []
        for data, changes in zip(minibatch,changepoints):
            self.add_data(data,changepoints=changes,generate=False,**kwargs)
            mb_states_list.append(self.states_list.pop())
        return mb_states_list

    def log_likelihood(self,data=None,changepoints=None,**kwargs):
        if data is not None:
            if isinstance(data,np.ndarray):
                assert isinstance(changepoints,list) or changepoints is None
                self.add_data(data=data,changepoints=changepoints,
                        generate=False,**kwargs)
                return self.states_list.pop().log_likelihood()
            else:
                assert isinstance(data,list) and (changepoints is None
                    or isinstance(changepoints,list) and len(changepoints) == len(data))
                changepoints = changepoints if changepoints is not None \
                        else [None]*len(data)

                loglike = 0.
                for d, c in zip(data,changepoints):
                    self.add_data(data=d,changepoints=c,generate=False,**kwargs)
                    loglike += self.states_list.pop().log_likelihood()
                return loglike
        else:
            return sum(s.log_likelihood() for s in self.states_list)


################
#  HMM models  #
################


class HMMSegHDP(_WeakLimitHDPMixin, _HMMSegSVI, _HMMSegMeanfield, _HMMSegGibbsSampling):
    _trans_class = transitions.DATruncHDPHMMTransitions
    _trans_conc_class = None

class HMMSegStickyHDP(_WeakLimitHDPMixin, _HMMSegSVI, _HMMSegMeanfield, _HMMSegGibbsSampling):
    _trans_class = transitions.DATruncStickyHDPHMMTransitions
        #_trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = None

class HMMSegHDPWeakLimit(_WeakLimitHDPMixin, _HMMSegSVI, _HMMSegMeanfield, _HMMSegGibbsSampling):
        #_trans_class = transitions.DATruncStickyHDPHMMTransitions
    _trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = None

class HMMSegStickyHDPWeakLimit(_WeakLimitHDPMixin, _HMMSegSVI, _HMMSegMeanfield, _HMMSegGibbsSampling):
        #_trans_class = transitions.DATruncStickyHDPHMMTransitions
    # _trans_class = transitions.WeakLimitHDPHMMTransitions
    # _trans_conc_class = None
    def __init__(self,obs_distns,
            kappa=None,alpha=None,gamma=None,trans_matrix=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):
        assert (None not in (alpha,gamma)) ^ \
                (None not in (alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0))
        if None not in (alpha,gamma):
            trans_distn = transitions.WeakLimitStickyHDPHMMTransitions(
                    num_states=len(obs_distns),
                    kappa=kappa,alpha=alpha,gamma=gamma,trans_matrix=trans_matrix)
        else:
            trans_distn = transitions.WeakLimitStickyHDPHMMTransitionsConc(
                    num_states=len(obs_distns),
                    kappa=kappa,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0,
                    trans_matrix=trans_matrix)
        super(HMMSegStickyHDPWeakLimit,self).__init__(
                obs_distns=obs_distns,trans_distn=trans_distn,**kwargs)


class HMMPython(_HMMGibbsSampling,_HMMSVI,_HMMMeanField,_HMMEM,
        _HMMViterbiEM):
    pass


class HMM(HMMPython):
    _states_class = hmm_states.HMMStatesPython


class WeakLimitHDPHMMPython(_WeakLimitHDPMixin,HMMPython):
    # NOTE: shouldn't really inherit EM or ViterbiEM, but it's convenient!
    _trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHMMTransitionsConc


class WeakLimitHDPHMM(_WeakLimitHDPMixin,HMM):
    _trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHMMTransitionsConc


class DATruncHDPHMMPython(_WeakLimitHDPMixin,HMMPython):
    # NOTE: weak limit mixin is poorly named; we just want its init method
    _trans_class = transitions.DATruncHDPHMMTransitions
    _trans_conc_class = None


class DATruncHDPHMM(_WeakLimitHDPMixin,HMM):
    _trans_class = transitions.DATruncHDPHMMTransitions
    _trans_conc_class = None


class WeakLimitStickyHDPHMM(WeakLimitHDPHMM):
    # TODO concentration resampling, too!
    def __init__(self,obs_distns,
            kappa=None,alpha=None,gamma=None,trans_matrix=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            **kwargs):
        assert (None not in (alpha,gamma)) ^ \
                (None not in (alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0))
        if None not in (alpha,gamma):
            trans_distn = transitions.WeakLimitStickyHDPHMMTransitions(
                    num_states=len(obs_distns),
                    kappa=kappa,alpha=alpha,gamma=gamma,trans_matrix=trans_matrix)
        else:
            trans_distn = transitions.WeakLimitStickyHDPHMMTransitionsConc(
                    num_states=len(obs_distns),
                    kappa=kappa,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0,
                    trans_matrix=trans_matrix)
        super(WeakLimitStickyHDPHMM,self).__init__(
                obs_distns=obs_distns,trans_distn=trans_distn,**kwargs)



class HMMPossibleChangepoints(_HMMPossibleChangepointsMixin,HMM):
    pass

#################
#  HSMM Mixins  #
#################


class _HSMMBase(_HMMBase):
    _states_class = hsmm_states.HSMMStatesPython
    _trans_class = transitions.HSMMTransitions
    _trans_conc_class = transitions.HSMMTransitionsConc
    # _init_steady_state_class = initial_state.HSMMSteadyState # TODO

    def __init__(self,dur_distns,**kwargs):
        self.dur_distns = dur_distns
        super(_HSMMBase,self).__init__(**kwargs)

    def add_data(self,data,stateseq=None,trunc=None,
            right_censoring=True,left_censoring=False,**kwargs):
        self.states_list.append(self._states_class(
            model=self,
            data=np.asarray(data),
            stateseq=stateseq,
            right_censoring=right_censoring,
            left_censoring=left_censoring,
            trunc=trunc,
            **kwargs))

    @property
    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) \
                + sum(d.num_parameters() for d in self.dur_distns) \
                + self.num_states**2 - self.num_states

#     def plot_durations(self,colors=None,states_objs=None):
#         if colors is None:
#             colors = self._get_colors()
#         if states_objs is None:
#             states_objs = self.states_list

#         cmap = cm.get_cmap()
#         used_states = self._get_used_states(states_objs)
#         for state,d in enumerate(self.dur_distns):
#             if state in used_states:
#                 d.plot(color=cmap(colors[state]),
#                         data=[s.durations[s.stateseq_norep == state]
#                             for s in states_objs])
#         plt.title('Durations')

#     def plot(self,color=None):
#         plt.gcf() #.set_size_inches((10,10))
#         colors = self._get_colors(self.states_list)

#         num_subfig_cols = len(self.states_list)
#         for subfig_idx,s in enumerate(self.states_list):
#             plt.subplot(3,num_subfig_cols,1+subfig_idx)
#             self.plot_observations(colors=colors,states_objs=[s])

#             plt.subplot(3,num_subfig_cols,1+num_subfig_cols+subfig_idx)
#             s.plot(colors_dict=colors)

#             plt.subplot(3,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
#             self.plot_durations(colors=colors,states_objs=[s])


class _HSMMGibbsSampling(_HSMMBase,_HMMGibbsSampling):
    @line_profiled
    def resample_parameters(self,**kwargs):
        self.resample_dur_distns()
        super(_HSMMGibbsSampling,self).resample_parameters(**kwargs)

    def resample_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.resample_with_censoring_and_truncation(
            data=
            [s.durations_censored[s.untrunc_slice][s.stateseq_norep[s.untrunc_slice] == state]
                for s in self.states_list],
            censored_data=
            [s.durations_censored[s.trunc_slice][s.stateseq_norep[s.trunc_slice] == state]
                for s in self.states_list])
        self._clear_caches()

    def copy_sample(self):
        new = super(_HSMMGibbsSampling,self).copy_sample()
        new.dur_distns = [d.copy_sample() for d in self.dur_distns]
        return new


class _HSMMEM(_HSMMBase,_HMMEM):
    def _M_step(self):
        super(_HSMMEM,self)._M_step()
        self._M_step_dur_distns()

    def _M_step_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [np.arange(1,s.expected_durations[state].shape[0]+1)
                        for s in self.states_list],
                    [s.expected_durations[state] for s in self.states_list])


class _HSMMMeanField(_HSMMBase,_HMMMeanField):
    def meanfield_update_parameters(self):
        super(_HSMMMeanField,self).meanfield_update_parameters()
        self.meanfield_update_dur_distns()

    def meanfield_update_dur_distns(self):
        for state, d in enumerate(self.dur_distns):
            d.meanfieldupdate(
                    [np.arange(1,s.expected_durations[state].shape[0]+1)
                        for s in self.states_list],
                    [s.expected_durations[state] for s in self.states_list])

    def _vlb(self):
        vlb = super(_HSMMMeanField,self)._vlb()
        vlb += sum(d.get_vlb() for d in self.dur_distns)
        return vlb


class _HSMMSVI(_HSMMBase,_HMMSVI):
    def _meanfield_sgdstep_parameters(self,mb_states_list,minibatchfrac,stepsize):
        super(_HSMMSVI,self)._meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_dur_distns(mb_states_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_dur_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, d in enumerate(self.dur_distns):
            d.meanfield_sgdstep(
                    [np.arange(1,s.expected_durations[state].shape[0]+1)
                        for s in mb_states_list],
                    [s.expected_durations[state] for s in mb_states_list],
                    minibatchfrac,stepsize)


class _HSMMINBEMMixin(_HMMEM,ModelEM):
    def EM_step(self):
        super(_HSMMINBEMMixin,self).EM_step()
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(data=None,stats=(
                sum(s.expected_dur_ns[state] for s in self.states_list),
                sum(s.expected_dur_tots[state] for s in self.states_list)))


class _HSMMViterbiEM(_HSMMBase,_HMMViterbiEM):
    def Viterbi_EM_step(self):
        super(_HSMMViterbiEM,self).Viterbi_EM_step()
        self._Viterbi_M_step_dur_distns()

    def _Viterbi_M_step_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])

    def _Viterbi_M_step_trans_distn(self):
        self.trans_distn.max_likelihood([s.stateseq_norep for s in self.states_list])


class _HSMMPossibleChangepointsMixin(_HMMPossibleChangepointsMixin):
    _states_class = hsmm_states.HSMMStatesPossibleChangepoints




class _DelayedMixin(object):
    def resample_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.resample_with_censoring_and_truncation(
            data=
            [s.durations_censored[s.untrunc_slice][s.stateseq_norep[s.untrunc_slice] == state]
                - s.delays[state] for s in self.states_list],
            censored_data=
            [s.durations_censored[s.trunc_slice][s.stateseq_norep[s.trunc_slice] == state]
                - s.delays[state] for s in self.states_list])
        self._clear_caches()

#################
#  HSMM Models  #
#################


class HSMMPython(_HSMMGibbsSampling,_HSMMSVI,_HSMMMeanField,
        _HSMMViterbiEM,_HSMMEM):
    _trans_class = transitions.HSMMTransitions
    _trans_conc_class = transitions.HSMMTransitionsConc


class HSMM(HSMMPython):
    _states_class = hsmm_states.HSMMStatesEigen


class GeoHSMM(HSMMPython):
    _states_class = hsmm_states.GeoHSMMStates


class DelayedGeoHSMM(_DelayedMixin,HSMMPython):
    _states_class = hsmm_states.DelayedGeoHSMMStates


class WeakLimitHDPHSMMPython(_WeakLimitHDPMixin,HSMMPython):
    # NOTE: shouldn't technically inherit EM or ViterbiEM, but it's convenient
    _trans_class = transitions.WeakLimitHDPHSMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHSMMTransitionsConc


class WeakLimitHDPHSMM(_WeakLimitHDPMixin,HSMM):
    _trans_class = transitions.WeakLimitHDPHSMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHSMMTransitionsConc


class WeakLimitGeoHDPHSMM(WeakLimitHDPHSMM):
    _states_class = hsmm_states.GeoHSMMStates

    def _M_step_dur_distns(self):
        warn('untested!')
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    stats=(
                        sum(s._expected_ns[state] for s in self.states_list),
                        sum(s._expected_tots[state] for s in self.states_list),
                        ))


class WeakLimitDelayedGeoHSMM(_DelayedMixin,WeakLimitHDPHSMM):
    _states_class = hsmm_states.DelayedGeoHSMMStates


class DATruncHDPHSMM(_WeakLimitHDPMixin,HSMM):
    # NOTE: weak limit mixin is poorly named; we just want its init method
    _trans_class = transitions.DATruncHDPHSMMTransitions
    _trans_conc_class = None



##########
#  meta  #
##########


class _SeparateTransMixin(object):
    def __init__(self,*args,**kwargs):
        super(_SeparateTransMixin,self).__init__(*args,**kwargs)
        self.trans_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.trans_distn))
        self.init_state_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.init_state_distn))

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct['trans_distns'] = dict(self.trans_distns.items())
        dct['init_state_distns'] = dict(self.init_state_distns.items())
        return dct

    def __setstate__(self,dct):
        self.__dict__.update(dct)
        self.trans_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.trans_distn))
        self.init_state_distns = collections.defaultdict(
                lambda: copy.deepcopy(self.init_state_distn))
        self.trans_distns.update(dct['trans_distns'])
        self.init_state_distns.update(dct['init_state_distns'])

    ### parallel tempering

    def swap_sample_with(self,other):
        self.trans_distns, other.trans_distns = self.trans_distns, other.trans_distns
        self.init_state_distns, other.init_state_distns = \
                other.init_state_distns, self.init_state_distns
        for d1, d2 in zip(self.init_state_distns.values(),other.init_state_distns.values()):
            d1.model = self
            d2.model = other
        super(_SeparateTransMixin,self).swap_sample_with(other)

    ### Gibbs sampling

    def resample_trans_distn(self):
        for group_id, trans_distn in self.trans_distns.iteritems():
            trans_distn.resample([s.stateseq for s in self.states_list
                if hash(s.group_id) == hash(group_id)])
        self._clear_caches()

    def resample_init_state_distn(self):
        for group_id, init_state_distn in self.init_state_distns.iteritems():
            init_state_distn.resample([s.stateseq[0] for s in self.states_list
                if hash(s.group_id) == hash(group_id)])
        self._clear_caches()

    ### Mean field

    def meanfield_update_trans_distn(self):
        for group_id, trans_distn in self.trans_distns.iteritems():
            states_list = [s for s in self.states_list if hash(s.group_id) == hash(group_id)]
            if len(states_list) > 0:
                trans_distn.meanfieldupdate([s.expected_transcounts for s in states_list])

    def meanfield_update_init_state_distn(self):
        for group_id, init_state_distn in self.init_state_distns.iteritems():
            states_list = [s for s in self.states_list if hash(s.group_id) == hash(group_id)]
            if len(states_list) > 0:
                init_state_distn.meanfieldupdate([s.expected_states[0] for s in states_list])

    def _vlb(self):
        vlb = 0.
        vlb += sum(s.get_vlb() for s in self.states_list)
        vlb += sum(trans_distn.get_vlb()
                for trans_distn in self.trans_distns.itervalues())
        vlb += sum(init_state_distn.get_vlb()
                for init_state_distn in self.init_state_distns.itervalues())
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb

    ### SVI

    def _meanfield_sgdstep_trans_distn(self,mb_states_list,minibatchfrac,stepsize):
        for group_id, trans_distn in self.trans_distns.iteritems():
            trans_distn.meanfield_sgdstep(
                    [s.expected_transcounts for s in mb_states_list
                        if hash(s.group_id) == hash(group_id)],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_init_state_distn(self,mb_states_list,minibatchfrac,stepsize):
        for group_id, init_state_distn in self.init_state_distns.iteritems():
            init_state_distn.meanfield_sgdstep(
                    [s.expected_states[0] for s in mb_states_list
                        if hash(s.group_id) == hash(group_id)],
                    minibatchfrac,stepsize)

    ### EM

    def EM_step(self):
        raise NotImplementedError

    ### Viterbi

    def Viterbi_EM_step(self):
        raise NotImplementedError


class HMMSeparateTrans(_SeparateTransMixin,HMM):
    _states_class = hmm_states.HMMStatesEigenSeparateTrans


class WeakLimitHDPHMMSeparateTrans(_SeparateTransMixin,WeakLimitHDPHMM):
    _states_class = hmm_states.HMMStatesEigenSeparateTrans





