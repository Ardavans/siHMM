from __future__ import division
import numpy as np
from numpy import newaxis as na
import abc
import copy

from pyhsmm.util.stats import sample_discrete
try:
    from pyhsmm.util.cstats import sample_markov
except ImportError:
    from pyhsmm.util.stats import sample_markov
from pyhsmm.util.general import rle
import time

######################
#  Mixins and bases  #
######################

class _StatesBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,model,T=None,data=None,stateseq=None,
            generate=True,initialize_from_prior=True):
        self.model = model

        self.T = T if T is not None else data.shape[0]
        self.data = data

        self.clear_caches()

        if stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
        elif generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def copy_sample(self,newmodel):
        new = copy.copy(self)
        new.clear_caches() # saves space, though may recompute later for likelihoods
        new.model = newmodel
        new.stateseq = self.stateseq.copy()
        return new

    _kwargs = {}  # used in subclasses for joblib stuff

    ### model properties

    @property
    def obs_distns(self):
        return self.model.obs_distns

    @property
    def trans_matrix(self):
        return self.model.trans_distn.trans_matrix

    @property
    def pi_0(self):
        return self.model.init_state_distn.pi_0

    @property
    def num_states(self):
        return self.model.num_states

    ### convenience properties

    @property
    def stateseq_norep(self):
        return rle(self.stateseq)[0]

    @property
    def durations(self):
        return rle(self.stateseq)[1]

    ### generation

    @abc.abstractmethod
    def generate_states(self):
        pass

    ### messages and likelihoods

    # some cached things depends on model parameters, so caches should be
    # cleared when the model changes (e.g. when parameters are updated)

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._normalizer = None

    @property
    def aBl(self):
        if self._aBl is None : #or self._aBl is not None:
            data = self.data

            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    @abc.abstractmethod
    def log_likelihood(self):
        pass

class _SeparateTransMixin(object):
    def __init__(self,group_id,**kwargs):
        assert not isinstance(group_id,np.ndarray)
        self.group_id = group_id
        self._kwargs = dict(self._kwargs,group_id=group_id)

        super(_SeparateTransMixin,self).__init__(**kwargs)

        # access these to be sure they're instantiated
        self.trans_matrix
        self.pi_0

    @property
    def trans_matrix(self):
        return self.model.trans_distns[self.group_id].trans_matrix

    @property
    def pi_0(self):
        return self.model.init_state_distns[self.group_id].pi_0

    @property
    def mf_trans_matrix(self):
        return np.maximum(
                self.model.trans_distns[self.group_id].exp_expected_log_trans_matrix,
                1e-3)

    @property
    def mf_pi_0(self):
        return self.model.init_state_distns[self.group_id].exp_expected_log_init_state_distn

class _PossibleChangepointsMixin(object):
    def __init__(self,model,data,changepoints=None,**kwargs):
        changepoints = changepoints if changepoints is not None \
                else [(t,t+1) for t in xrange(data.shape[0])]

        self.changepoints = changepoints
        self.segmentstarts = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.segmentlens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)

        assert all(l > 0 for l in self.segmentlens)
        assert sum(self.segmentlens) == data.shape[0]
        assert self.changepoints[0][0] == 0 and self.changepoints[-1][-1] == data.shape[0]

        self._kwargs = dict(self._kwargs,changepoints=changepoints)

        super(_PossibleChangepointsMixin,self).__init__(
                model,T=len(changepoints),data=data,**kwargs)

    def clear_caches(self):
        self._aBBl = self._mf_aBBl = None
        self._stateseq = None
        super(_PossibleChangepointsMixin,self).clear_caches()

    @property
    def Tblock(self):
        return len(self.changepoints)

    @property
    def Tfull(self):
        return self.data.shape[0]

    @property
    def stateseq(self):
        if self._stateseq is None:
            self._stateseq = self.blockstateseq.repeat(self.segmentlens)
        return self._stateseq

    @stateseq.setter
    def stateseq(self,stateseq):
        assert len(stateseq) == self.Tblock or len(stateseq) == self.Tfull
        if len(stateseq) == self.Tblock:
            self.blockstateseq = stateseq
        else:
            self.blockstateseq = stateseq[self.segmentstarts]
        self._stateseq = None

    def _expected_states(self,*args,**kwargs):
        expected_states = \
            super(_PossibleChangepointsMixin,self)._expected_states(*args,**kwargs)
        return expected_states.repeat(self.segmentlens,axis=0)

    @property
    def aBl(self):
        if self._aBBl is None:
            aBl = super(_PossibleChangepointsMixin,self).aBl
            aBBl = self._aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._aBBl

    @property
    def mf_aBl(self):
        if self._mf_aBBl is None:
            aBl = super(_PossibleChangepointsMixin,self).mf_aBl
            aBBl = self._mf_aBBl = np.empty((self.Tblock,self.num_states))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._mf_aBBl

    def plot(self,*args,**kwargs):
        from matplotlib import pyplot as plt
        super(_PossibleChangepointsMixin,self).plot(*args,**kwargs)
        plt.xlim((0,self.Tfull))

    # TODO do generate() and generate_states() actually work?

####################
#  States classes  #
####################

class HMMStatesPython(_StatesBase):
    ### generation

    def generate_states(self):
        T = self.T
        nextstate_distn = self.pi_0
        A = self.trans_matrix

        stateseq = np.zeros(T,dtype=np.int32)
        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[stateseq[idx]]

        self.stateseq = stateseq
        return stateseq

    ### message passing

    def log_likelihood(self):
        if self._normalizer is None:
            self.messages_forwards_normalized() # NOTE: sets self._normalizer
        return self._normalizer

    def _messages_log(self,trans_matrix,init_state_distn,log_likelihoods):
        alphal = self._messages_forwards_log(trans_matrix,init_state_distn,log_likelihoods)
        betal = self._messages_backwards_log(trans_matrix,log_likelihoods)
        return alphal, betal

    def messages_log(self):
        return self._messages_log(self.trans_matrix,self.pi_0,self.aBl)

    @staticmethod
    def _messages_backwards_log1(trans_matrix,log_likelihoods):
        errs = np.seterr(over='ignore')
        Al = np.log(trans_matrix)
        aBl = log_likelihoods

        betal = np.zeros_like(aBl)

        for t in xrange(betal.shape[0]-2,-1,-1):
            np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])

        np.seterr(**errs)
        return betal

    def messages_backwards_log(self):
        betal = self._messages_backwards_log(self.trans_matrix,self.aBl)
        assert not np.isnan(betal).any()
        self._normalizer = np.logaddexp.reduce(np.log(self.pi_0) + betal[0] + self.aBl[0])
        return betal

    @staticmethod
    def _messages_forwards_log1(trans_matrix,init_state_distn,log_likelihoods):
        errs = np.seterr(over='ignore')
        Al = np.log(trans_matrix)
        aBl = log_likelihoods

        alphal = np.zeros_like(aBl)

        alphal[0] = np.log(init_state_distn) + aBl[0]
        for t in xrange(alphal.shape[0]-1):
            alphal[t+1] = np.logaddexp.reduce(alphal[t] + Al.T,axis=1) + aBl[t+1]

        np.seterr(**errs)
        return alphal

    def messages_forwards_log(self):
        alphal = self._messages_forwards_log(self.trans_matrix,self.pi_0,self.aBl)
        assert not np.any(np.isnan(alphal))
        self._normalizer = np.logaddexp.reduce(alphal[-1])
        return alphal

    @staticmethod
    def _messages_backwards_normalized(trans_matrix,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        A = trans_matrix
        T = aBl.shape[0]

        betan = np.empty_like(aBl)
        logtot = 0.

        betan[-1] = 1.
        for t in xrange(T-2,-1,-1):
            cmax = aBl[t+1].max()
            betan[t] = A.dot(betan[t+1] * np.exp(aBl[t+1] - cmax))
            norm = betan[t].sum()
            logtot += cmax + np.log(norm)
            betan[t] /= norm

        cmax = aBl[0].max()
        logtot += cmax + np.log((np.exp(aBl[0] - cmax) * init_state_distn * betan[0]).sum())

        return betan, logtot

    def messages_backwards_normalized(self):
        betan, self._normalizer = \
                self._messages_backwards_normalized(self.trans_matrix,self.pi_0,self.aBl)
        return betan

    @staticmethod
    def _messages_forwards_normalized(trans_matrix,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        A = trans_matrix
        T = aBl.shape[0]

        alphan = np.empty_like(aBl)
        logtot = 0.

        in_potential = init_state_distn
        for t in xrange(T):
            cmax = aBl[t].max()
            alphan[t] = in_potential * np.exp(aBl[t] - cmax)
            norm = alphan[t].sum()
            if norm != 0:
                alphan[t] /= norm
                logtot += np.log(norm) + cmax
            else:
                alphan[t:] = 0.
                return alphan, np.log(0.)
            in_potential = alphan[t].dot(A)

        return alphan, logtot

    def messages_forwards_normalized(self):
        alphan, self._normalizer = \
                self._messages_forwards_normalized(self.trans_matrix,self.pi_0,self.aBl)
        return alphan

    ### Gibbs sampling

    def resample_log(self):
        betal = self.messages_backwards_log()
        self.sample_forwards_log(betal)

    def resample_normalized(self):
        alphan = self.messages_forwards_normalized()
        self.sample_backwards_normalized(alphan)

    def resample(self):
        return self.resample_normalized()

    @staticmethod
    def _sample_forwards_log(betal,trans_matrix,init_state_distn,log_likelihoods):
        A = trans_matrix
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        nextstate_unsmoothed = init_state_distn
        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf
            if np.any(np.isfinite(logdomain)):
                stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            else:
                stateseq[idx] = sample_discrete(nextstate_unsmoothed)
            nextstate_unsmoothed = A[stateseq[idx]]

        return stateseq

    def sample_forwards_log(self,betal):
        self.stateseq = self._sample_forwards_log(betal,self.trans_matrix,self.pi_0,self.aBl)

    @staticmethod
    def _sample_forwards_normalized(betan,trans_matrix,init_state_distn,log_likelihoods):
        A = trans_matrix
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        nextstate_unsmoothed = init_state_distn
        for idx in xrange(T):
            logdomain = aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf
            stateseq[idx] = sample_discrete(nextstate_unsmoothed * betan * np.exp(logdomain - np.amax(logdomain)))
            nextstate_unsmoothed = A[stateseq[idx]]

        return stateseq

    def sample_forwards_normalized(self,betan):
        self.stateseq = self._sample_forwards_normalized(
                betan,self.trans_matrix,self.pi_0,self.aBl)

    @staticmethod
    def _sample_backwards_normalized(alphan,trans_matrix_transpose):
        AT = trans_matrix_transpose
        T = alphan.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        next_potential = np.ones(AT.shape[0])
        for t in xrange(T-1,-1,-1):
            stateseq[t] = sample_discrete(next_potential * alphan[t])
            next_potential = AT[stateseq[t]]

        return stateseq

    def sample_backwards_normalized(self,alphan):
        self.stateseq = self._sample_backwards_normalized(alphan,self.trans_matrix.T.copy())

    ### Mean Field

    @property
    def mf_aBl(self):
        if self._mf_aBl is None:
            T = self.data.shape[0]
            self._mf_aBl = aBl = np.empty((T,self.num_states))

            for idx, o in enumerate(self.obs_distns):
                aBl[:,idx] = o.expected_log_likelihood(self.data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._mf_aBl

    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix
        # return np.maximum(self.model.trans_distn.exp_expected_log_trans_matrix,1e-5)

    @property
    def mf_pi_0(self):
        return self.model.init_state_distn.exp_expected_log_init_state_distn

    @property
    def all_expected_stats(self):
        return self.expected_states, self.expected_transcounts, self._normalizer

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self.expected_transcounts, self._normalizer = vals
        self.stateseq = self.expected_states.argmax(1).astype('int32') # for plotting

    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_matrix,self.mf_pi_0,self.mf_aBl)

    def get_vlb(self):
        if self._normalizer is None:
            self.meanfieldupdate() # NOTE: sets self._normalizer
        return self._normalizer

    def _expected_statistics(self,trans_potential,init_potential,likelihood_log_potential):
        tic = time.time()
        alphal = self._messages_forwards_log1(trans_potential,init_potential,
                likelihood_log_potential)
        betal = self._messages_backwards_log1(trans_potential,likelihood_log_potential)
        #print time.time() - tic
        expected_states, expected_transcounts, normalizer = \
                self._expected_statistics_from_messages_slow(trans_potential,likelihood_log_potential,alphal,betal)
        assert not np.isinf(expected_states).any()
        return expected_states, expected_transcounts, normalizer

    @staticmethod
    def _expected_statistics_from_messages_slow(trans_potential,likelihood_log_potential,alphal,betal):
        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:,na]
        np.exp(expected_states,out=expected_states)
        expected_states /= expected_states.sum(1)[:,na]

        Al = np.log(trans_potential)
        log_joints = alphal[:-1,:,na] + (betal[1:,na,:] + likelihood_log_potential[1:,na,:]) + Al[na,...]
        log_joints -= log_joints.max((1,2))[:,na,na]
        joints = np.exp(log_joints)
        joints /= joints.sum((1,2))[:,na,na] # NOTE: renormalizing each isnt really necessary
        expected_transcounts = joints.sum(0)

        normalizer = np.logaddexp.reduce(alphal[0] + betal[0])

        return expected_states, expected_transcounts, normalizer

    ### EM

    def E_step(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.trans_matrix,self.pi_0,self.aBl)

    ### Viterbi

    def Viterbi(self):
        scores, args = self.maxsum_messages_backwards()
        self.maximize_forwards(scores,args)

    def maxsum_messages_backwards(self):
        return self._maxsum_messages_backwards(self.trans_matrix,self.aBl)

    def maximize_forwards(self,scores,args):
        self.stateseq = self._maximize_forwards(scores,args,self.pi_0,self.aBl)


    def mf_Viterbi(self):
        scores, args = self.mf_maxsum_messages_backwards()
        self.mf_maximize_forwards(scores,args)

    def mf_maxsum_messages_backwards(self):
        return self._maxsum_messages_backwards(self.mf_trans_matrix,self.mf_aBl)

    def mf_maximize_forwards(self,scores,args):
        self.stateseq = self._maximize_forwards(scores,args,self.mf_pi_0,self.mf_aBl)


    @staticmethod
    def _maxsum_messages_backwards(trans_matrix, log_likelihoods):
        errs = np.seterr(divide='ignore')
        Al = np.log(trans_matrix)
        np.seterr(**errs)
        aBl = log_likelihoods

        scores = np.zeros_like(aBl)
        args = np.zeros(aBl.shape,dtype=np.int32)

        for t in xrange(scores.shape[0]-2,-1,-1):
            vals = Al + scores[t+1] + aBl[t+1]
            vals.argmax(axis=1,out=args[t+1])
            vals.max(axis=1,out=scores[t])

        return scores, args

    @staticmethod
    def _maximize_forwards(scores,args,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        stateseq[0] = (scores[0] + np.log(init_state_distn) + aBl[0]).argmax()
        for idx in xrange(1,T):
            stateseq[idx] = args[idx,stateseq[idx-1]]

        return stateseq


################ HDP model


class _RobustDPStatesBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,model,T=None, data=None,stateseq=None,
            generate=True,initialize_from_prior=True):
        self.model = model

        self.T = T if T is not None else data.shape[0] #num of words
        self.data = data

        self.clear_caches()

        if stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
        elif generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def copy_sample(self,newmodel):
        new = copy.copy(self)
        new.clear_caches() # saves space, though may recompute later for likelihoods
        new.model = newmodel
        new.stateseq = self.stateseq.copy()
        return new

    _kwargs = {}  # used in subclasses for joblib stuff

    ### model properties

    @property
    def obs_distns(self):
        return self.model.obs_distns

    @property
    def trans_matrix(self):
        return self.model.trans_distn.trans_matrix


    @property
    def num_states(self):
        return self.model.num_states

    ### convenience properties

    @property
    def stateseq_norep(self):
        return rle(self.stateseq)[0]


    ### generation

    @abc.abstractmethod
    def generate_states(self):
        pass

    ### messages and likelihoods

    # some cached things depends on model parameters, so caches should be
    # cleared when the model changes (e.g. when parameters are updated)

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._normalizer = None

    @property
    def aBl(self):
        if self._aBl is None : #or self._aBl is not None:
            data = self.data

            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.

        return self._aBl

    @abc.abstractmethod
    def log_likelihood(self):
        pass


class HDPStates(_RobustDPStatesBase):
    ### generation

    def generate_states(self):
        T = self.T
        doc_num = self.doc_num
        state_distn = self.trans_matrix[doc_num, :]

        stateseq = np.zeros(T,dtype=np.int32)
        for idx in xrange(T):
            stateseq[idx] = sample_discrete(state_distn)

        self.stateseq = stateseq
        return stateseq

    ### message passing

    def log_likelihood(self):
        if self._normalizer is None:
            self.messages_forwards_log() # NOTE: sets self._normalizer
        return self._normalizer

    # def _messages_log(self,trans_matrix, log_likelihoods):
    #     alphal = self._messages_forwards_log(trans_matrix[self.doc_num, :], log_likelihoods)
    #     return alphal
    #
    # def messages_log(self):
    #     return self._messages_log(self.trans_matrix, self.aBl)


    @staticmethod
    def _messages_forwards_log(trans_matrix_docnum, log_likelihoods):
        errs = np.seterr(over='ignore')
        aBl = log_likelihoods
        np.seterr(**errs)
        return aBl + trans_matrix_docnum

    def messages_forwards_log(self):
        alphal = self._messages_forwards_log(self.trans_matrix[self.doc_num, :], self.mf_aBl)
        assert not np.any(np.isnan(alphal))
        self._normalizer = np.logaddexp.reduce(np.logaddexp.reduce(alphal, 1)) #TODO check this: we should marginalize over two dimensions
        return alphal



    ### Gibbs sampling

    def resample_log(self):
        #betal = self.messages_backwards_log()
        self.sample_forwards_log()


    def resample(self):
        return self.resample_log()

    @staticmethod
    def _sample_forwards_log(trans_matrix_docnum, log_likelihoods):
        A = trans_matrix_docnum
        aBl = log_likelihoods
        T = aBl.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        nextstate_unsmoothed = trans_matrix_docnum
        for idx in xrange(T):
            logdomain = aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf
            if np.any(np.isfinite(logdomain)):
                stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            else:
                stateseq[idx] = sample_discrete(nextstate_unsmoothed)
            nextstate_unsmoothed = trans_matrix_docnum

        return stateseq

    def sample_forwards_log(self,betal):
        self.stateseq = self._sample_forwards_log(self.trans_matrix[self.doc_num],self.aBl)


    ### Mean Field

    @property
    def mf_aBl(self):
        if self._mf_aBl is None:
            T = self.data.shape[0]
            self._mf_aBl = aBl = np.empty((T,self.num_states))

            for idx, o in enumerate(self.obs_distns):
                #aBl[:,idx] = o.expected_log_likelihood(self.data).ravel()
                aBl[:,idx] = o.expected_log_likelihood([i[0] for i in self.data]).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.


        ####temp for debugging
        # import matplotlib.pylab as plt
        # distance_length = np.sqrt(np.sum((self.data[1,:] - self.data[:,:])**2, 1))
        # log_likelihood_val = self._mf_aBl[:,np.argmax(self._mf_aBl[1,:])]
        # plt.plot(distance_length, log_likelihood_val, '.')
        # plt.title('doc num: ' + str(self.doc_num))
        # plt.xlabel('L2 distance')
        # plt.ylabel('log likelihood')
        # plt.show()

        return self._mf_aBl

    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix
        # return np.maximum(self.model.trans_distn.exp_expected_log_trans_matrix,1e-5)



    @property
    def all_expected_stats(self):
        return self.expected_states, self._normalizer

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self._normalizer = vals
        self.stateseq = self.expected_states.argmax(1).astype('int32') # for plotting

    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_matrix[self.doc_num,:],self.mf_aBl)

    def get_vlb(self):
        if self._normalizer is None:
            self.meanfieldupdate() # NOTE: sets self._normalizer
        return self._normalizer

    def _expected_statistics(self,trans_potential_docnum,likelihood_log_potential):
        alphal = self._messages_forwards_log(trans_potential_docnum, likelihood_log_potential)

        expected_states, normalizer = \
                self._expected_statistics_from_messages_slow(alphal)
        assert not np.isinf(expected_states).any()
        return expected_states, normalizer

    @staticmethod
    def _expected_statistics_from_messages_slow(alphal):
        expected_states = alphal
        expected_states -= expected_states.max(1)[:,na]
        np.exp(expected_states,out=expected_states)
        expected_states /= expected_states.sum(1)[:,na]


        normalizer = np.logaddexp.reduce(np.logaddexp.reduce(alphal, 1)) #TODO check this

        return expected_states, normalizer




# The class that we are using for the nonparametric segmentation model
class HMMSegExStatesEigen(HMMStatesPython):

    def __init__(self, window_data = None, *args, **kwargs):
        super(HMMSegExStatesEigen, self).__init__(*args, **kwargs)
        self.window_data = np.array(window_data)

    def clear_caches(self):
        self._aBl = self._mf_aBl = None
        self._normalizer = None
        self._loglinearterm_aBl = None

    def log_likelihood(self):
        #self.clear_caches()
        if self._normalizer is None:
            if self.model.svi_or_gibbs == 'svi':
                alpha = self._messages_forwards_log_fast(self.mf_trans_matrix,
                                                 self.mf_pi_0,
                                                 self.mf_llt) # NOTE: sets self._normalizer
            elif self.model.svi_or_gibbs == 'gibbs':
                alpha = self._messages_forwards_log_fast(self.trans_matrix,
                                                 self.pi_0,
                                                 self.llt)
            return np.logaddexp.reduce(alpha[-1])

    # This is for Gibbs (non synthetic dataset)
    # def generate_states(self):
    #     T = self.T
    #     nhs = self.num_states
    #     nextstate_distn = self.pi_0
    #     A = self.trans_matrix
    #
    #     #print 'hereeeeeeeeeeeeee'
    #     # A = np.array([[0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     #                  [0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     #                  [0.3, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     #                  [0.0, 0.0, 0.0, 0.7, 0.15, 0.15, 0.0, 0.0, 0.0],
    #     #                  [0.0, 0.0, 0.0, 0.15, 0.7, 0.15, 0.0, 0.0, 0.0],
    #     #                  [0.0, 0.0, 0.0, 0.15, 0.15, 0.7, 0.0, 0.0, 0.0],
    #     #                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1],
    #     #                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
    #     #                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8]])
    #     #
    #     # A = np.array([[1, 0],
    #     #                  [0, 1]])
    #
    #     #s_distn = np.array([0.1, 0.9]) # 0)prob of staying in the regime , 1) prob of starting a new one
    #     # s_distn_0 = 0.05 #probability of a new segment
    #     # s_distn = np.array([s_distn_0, 1 - s_distn_0])
    #
    #     w_vec = self.feature_weights
    #     stateseq = np.zeros(T,dtype=np.int32)
    #     true_segmentation = np.ones(T, dtype=int)
    #     true_segmentation_prob = np.zeros(T)
    #     prev_segment_val = 0
    #     prev_group = -1
    #     for idx in xrange(T):
    #         stateseq[idx] = sample_discrete(nextstate_distn)
    #         if prev_segment_val == 1:
    #             stateseq[idx] += nhs
    #
    #         # current_group = int(stateseq[idx] / 3) #9 states
    #         # #current_group = int(stateseq[idx] / 1) #3 states
    #         # if current_group != prev_group and prev_group != -1:
    #         #     true_segmentation[idx] = 1
    #         # prev_group = current_group
    #
    #         if self.model.bern_or_weight == 'weight':
    #             temp_state_indicator = np.zeros(nhs)
    #             temp_state_indicator[stateseq[idx]] = 1
    #             s_distn_0 = np.exp(0 -
    #                              np.logaddexp(0, (np.sum(w_vec[-nhs-1:-1] * temp_state_indicator) + w_vec[-1] )))
    #             s_distn = np.array([s_distn_0, 1 - s_distn_0])
    #         else:
    #             temp_current_state = stateseq[idx] - nhs if stateseq[idx] >= nhs else stateseq[idx]
    #             s_distn = self.model.segmentation_distns[temp_current_state].weights
    #         print 'logsdistn_0', np.log(s_distn_0)
    #
    #         s_state = sample_discrete(s_distn)
    #         if s_state == 1:
    #             nextstate_distn = A[temp_current_state]
    #         else:
    #             true_segmentation[idx] = 0
    #
    #             nextstate_distn = self.pi_0
    #         prev_segment_val = true_segmentation[idx]
    #         true_segmentation_prob[idx] = s_distn[0]
    #     self.stateseq = stateseq
    #     self.true_segmentation = true_segmentation
    #     self.true_segmentation_prob = true_segmentation_prob
    #     return stateseq

    #This is used for synth_experiment.py with 9 hidden states (not the illustrative one)
    def generate_states(self):
        T = self.T
        nhs = self.num_states
        nextstate_distn = self.pi_0
        A = self.trans_matrix

        #print 'hereeeeeeeeeeeeee'
        A = np.array([[0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.3, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.7, 0.15, 0.15, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.15, 0.7, 0.15, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.15, 0.15, 0.7, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.1],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8]])

        # A = np.array([[1, 0],
        #                  [0, 1]])

        #s_distn = np.array([0.1, 0.9]) # 0)prob of staying in the regime , 1) prob of starting a new one
        s_distn_0 = 0.05 #probability of a new segment
        s_distn = np.array([s_distn_0, 1 - s_distn_0])

        w_vec = self.feature_weights
        stateseq = np.zeros(T,dtype=np.int32)
        true_segmentation = np.ones(T, dtype=int)
        true_segmentation_prob = np.zeros(T)
        prev_segment_val = 0
        prev_group = -1
        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            # if prev_segment_val == 1:
            #     stateseq[idx] += nhs

            #onlly for the 9 state matrix also comment the line after nextstate_distn = self.pi_0 also the matrix A above
            current_group = int(stateseq[idx] / 3) #9 states
            #current_group = int(stateseq[idx] / 1) #3 states
            if current_group != prev_group and prev_group != -1:
                true_segmentation[idx - 1] = 0
            prev_group = current_group

            # if self.model.bern_or_weight == 'weight':
            #     temp_state_indicator = np.zeros(nhs)
            #     temp_state_indicator[stateseq[idx]] = 1
            #     s_distn_0 = np.exp(0 -
            #                      np.logaddexp(0, (np.sum(w_vec[-nhs-1:-1] * temp_state_indicator) + w_vec[-1] )))
            #     s_distn = np.array([s_distn_0, 1 - s_distn_0])
            # else:
            #     temp_current_state = stateseq[idx] - nhs if stateseq[idx] >= nhs else stateseq[idx]
            #     s_distn = self.model.segmentation_distns[temp_current_state].weights
            #print 'logsdistn_0', np.log(s_distn_0)

            s_state = sample_discrete(s_distn)
            if s_state == 1:
                nextstate_distn = A[stateseq[idx]]
            else:
                nextstate_distn = self.pi_0
                # true_segmentation[idx] = 0

            true_segmentation_prob[idx] = s_distn[0]
        self.stateseq = stateseq
        self.true_segmentation = true_segmentation
        self.true_segmentation_prob = true_segmentation_prob

        return stateseq

    #this is for the illustrative example with 8 states
    # def generate_states(self):
    #     T = self.T
    #     nhs = self.num_states
    #     nextstate_distn = self.pi_0
    #     A = self.trans_matrix
    #
    #     #print 'hereeeeeeeeeeeeee'
    #     A = np.array([[0.05, 0.8, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0],
    #                      [0.05, 0.05, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
    #                      [0.05, 0.05, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0],
    #                      [0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
    #                      [0.0, 0.0, 0.0, 0.0, 0.85, 0.05, 0.05, 0.05],
    #                      [0.0, 0.0, 0.0, 0.0, 0.05, 0.85, 0.05, 0.05],
    #                      [0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.85, 0.05],
    #                      [0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.85]])
    #
    #     # A = np.array([[1, 0],
    #     #                  [0, 1]])
    #
    #     #s_distn = np.array([0.1, 0.9]) # 0)prob of staying in the regime , 1) prob of starting a new one
    #     s_distn_0 = 0.05 #probability of a new segment
    #     s_distn = np.array([s_distn_0, 1 - s_distn_0])
    #
    #     w_vec = self.feature_weights
    #     stateseq = np.zeros(T,dtype=np.int32)
    #     true_segmentation = np.ones(T, dtype=int)
    #     true_segmentation_prob = np.zeros(T)
    #     prev_segment_val = 0
    #     prev_group = -1
    #     for idx in xrange(T):
    #         stateseq[idx] = sample_discrete(nextstate_distn)
    #         # if prev_segment_val == 1:
    #         #     stateseq[idx] += nhs
    #
    #         #onlly for the 9 state matrix also comment the line after nextstate_distn = self.pi_0 also the matrix A above
    #         current_group = int(stateseq[idx] / 4) #9 states
    #         #current_group = int(stateseq[idx] / 1) #3 states
    #         if current_group != prev_group and prev_group != -1:
    #             true_segmentation[idx - 1] = 0
    #         prev_group = current_group
    #
    #         # if self.model.bern_or_weight == 'weight':
    #         #     temp_state_indicator = np.zeros(nhs)
    #         #     temp_state_indicator[stateseq[idx]] = 1
    #         #     s_distn_0 = np.exp(0 -
    #         #                      np.logaddexp(0, (np.sum(w_vec[-nhs-1:-1] * temp_state_indicator) + w_vec[-1] )))
    #         #     s_distn = np.array([s_distn_0, 1 - s_distn_0])
    #         # else:
    #         #     temp_current_state = stateseq[idx] - nhs if stateseq[idx] >= nhs else stateseq[idx]
    #         #     s_distn = self.model.segmentation_distns[temp_current_state].weights
    #         #print 'logsdistn_0', np.log(s_distn_0)
    #
    #         s_state = sample_discrete(s_distn)
    #         if s_state == 1:
    #             nextstate_distn = A[stateseq[idx]]
    #         else:
    #             nextstate_distn = self.pi_0
    #             # true_segmentation[idx] = 0
    #
    #         true_segmentation_prob[idx] = s_distn[0]
    #     self.stateseq = stateseq
    #     self.true_segmentation = true_segmentation
    #     self.true_segmentation_prob = true_segmentation_prob
    #
    #     return stateseq

    # @property
    # def loglinearterm_aBl(self):
    #     if self._loglinearterm_aBl is None : #or self._aBl is not None:
    #         data = self.data
    #
    #         aBl = np.empty((data.shape[0],self.num_states))
    #         for idx, obs_distn in enumerate(self.obs_distns):
    #             aBl[:,idx] = obs_distn.log_likelihood(data).ravel()
    #         aBl[np.isnan(aBl).any(1)] = 0.
    #         temp_aBl = np.hstack((aBl, aBl))
    #
    #
    #         feature_weights = self.feature_weights
    #         window_data = self.window_data
    #         nhs = self.num_states
    #
    #         temp_constant = np.sum(feature_weights[:-nhs-1] * window_data[:,:], axis = 1) + feature_weights[-1]
    #         temp_exp = temp_constant[:, None] + feature_weights[-nhs-1:-1]
    #         temp_logaddexp = np.logaddexp(0, temp_exp)
    #         loglinearterm =  np.tile(temp_exp, 2) * np.repeat([0,1], nhs) - np.tile(temp_logaddexp, 2)
    #
    #         self._loglinearterm_aBl = temp_aBl+ loglinearterm
    #     return self._loglinearterm_aBl


    @property
    def feature_weights(self):
        return self.model.feature_weights



    @property
    def all_expected_stats(self):
        return self.expected_initial, self.expected_states, self.expected_transcounts_segmentation, self._normalizer, \
               self.expected_joints

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_initial, self.expected_states, self.expected_transcounts_segmentation, self._normalizer,\
        self.expected_joints = vals

    ### messages for the forward backward

    def E_step(self):
        self.clear_caches()
        # self.all_expected_stats = self._expected_statistics(
        #         self.trans_matrix,self.pi_0,self.aBl, self.feature_weights)
        self.all_expected_stats = self._expected_statistics(
                self.trans_matrix,self.pi_0, self.loglinearterm_aBl)

    @staticmethod
    def _messages_forwards_log_slow(trans_potential, init_potential, likelihood_log_potential,
                                     feature_weights, window_data):

        errs = np.seterr(over='ignore')
        Al = np.log(trans_potential)
        aBl = likelihood_log_potential
        sequence_length = aBl.shape[0]
        pil = np.log(init_potential)
        nhs = trans_potential.shape[0]
        alphal = np.zeros((sequence_length, nhs * 2)) #the first num_states columns coorepond to s_t = 0 and the rest is for s_t = 1

        temp_constant = np.sum(feature_weights[:-nhs-1] * window_data[0,:]) + feature_weights[-1]
        temp_exp = temp_constant + feature_weights[-nhs-1:-1]
        temp_logaddexp = np.logaddexp(0, temp_exp)
        temp_log_linear = np.tile(temp_exp, 2) * np.repeat([0,1], nhs) - np.tile(temp_logaddexp, 2)
        # alphal[0] = np.tile(np.log(init_potential), 2) + np.tile(aBl[0], 2) + temp_log_linear
        alphal[0] = np.tile(np.log(init_potential), 2) + np.hstack((aBl[0], aBl[0])) + temp_log_linear
        giant_Al_pil = np.tile(np.vstack((np.tile(pil, (nhs,1)), Al )), (1,2))
        for t in xrange(alphal.shape[0]-1):
            temp_constant = np.sum(feature_weights[:-nhs-1] * window_data[t+1,:]) + feature_weights[-1]
            temp_exp = temp_constant + feature_weights[-nhs-1:-1]
            temp_logaddexp = np.logaddexp(0, temp_exp)
            temp_log_linear = np.tile(temp_exp, 2) * np.repeat([0,1], nhs) - np.tile(temp_logaddexp, 2)


            alphal[t+1] = np.logaddexp.reduce(alphal[t][:,None] + giant_Al_pil,axis=0) + \
                          np.hstack((aBl[t+1], aBl[t+1])) + temp_log_linear


        np.seterr(**errs)
        return alphal



    @staticmethod
    def _messages_backwards_log_slow(trans_potential, init_potential, likelihood_log_potential,
                                     feature_weights, window_data):
        errs = np.seterr(over='ignore')
        Al = np.log(trans_potential)
        pil = np.log(init_potential)
        aBl = likelihood_log_potential
        nhs = trans_potential.shape[0]
        sequence_length = aBl.shape[0]
        betal = np.zeros((sequence_length, nhs * 2))
        giant_Al_pil = np.tile(np.vstack((np.tile(pil, (nhs,1)), Al )), (1,2))
        for t in xrange(betal.shape[0]-2,-1,-1):
            temp_constant = np.sum(feature_weights[:-nhs-1] * window_data[t+1,:]) + feature_weights[-1]
            temp_exp = temp_constant + feature_weights[-nhs-1:-1]
            temp_logaddexp = np.logaddexp(0, temp_exp)
            temp_log_linear = np.tile(temp_exp, 2) * np.repeat([0,1], nhs) - np.tile(temp_logaddexp, 2)

            np.logaddexp.reduce( giant_Al_pil + betal[t+1] +
                                 np.hstack((aBl[t+1], aBl[t+1])) +
                                 temp_log_linear
                                ,axis=1 ,out=(betal[t]))


        np.seterr(**errs)
        return betal


    @staticmethod
    def _messages_forwards_log_fast(trans_potential, init_potential, likelihood_log_potential_llt):

        errs = np.seterr(over='ignore')
        Al = np.log(trans_potential)
        aBl = likelihood_log_potential_llt
        sequence_length = aBl.shape[0]
        pil = np.log(init_potential)
        nhs = trans_potential.shape[0]
        alphal = np.zeros((sequence_length, nhs * 2)) #the first num_states columns coorepond to s_t = 0 and the rest is for s_t = 1


        alphal[0] = np.tile(np.log(init_potential), 2) + aBl[0]
        giant_Al_pil = np.tile(np.vstack((np.tile(pil, (nhs,1)), Al )), (1,2))
        for t in xrange(alphal.shape[0]-1):

            alphal[t+1] = np.logaddexp.reduce(alphal[t][:,None] + giant_Al_pil,axis=0) + aBl[t+1]


        np.seterr(**errs)
        return alphal



    @staticmethod
    def _messages_backwards_log_fast(trans_potential, init_potential, likelihood_log_potential_llt):
        errs = np.seterr(over='ignore')
        Al = np.log(trans_potential)
        pil = np.log(init_potential)
        aBl = likelihood_log_potential_llt
        nhs = trans_potential.shape[0]
        sequence_length = aBl.shape[0]
        betal = np.zeros((sequence_length, nhs * 2))
        giant_Al_pil = np.tile(np.vstack((np.tile(pil, (nhs,1)), Al )), (1,2))


        for t in xrange(betal.shape[0]-2,-1,-1):
            np.logaddexp.reduce( giant_Al_pil + betal[t+1] + aBl[t+1], axis=1, out=(betal[t]))

        np.seterr(**errs)
        return betal


    ### Gibbs sampling
    def resample_log(self):
        betal = self._messages_backwards_log_fast(self.trans_matrix,self.pi_0,self.llt)
        self.sample_forwards_log(betal)

    def resample(self):
        return self.resample_log()

    def sample_forwards_log(self, betal):
        self.stateseq, self.true_segmentation = self._sample_forwards_log(betal,self.trans_matrix,self.pi_0,self.llt)

    def _sample_forwards_log(self, betal, trans_matrix, init_state_distn, log_likelihoods_loglinear):
        errs = np.seterr(over='ignore')
        Al = trans_matrix
        aBl = log_likelihoods_loglinear
        T = aBl.shape[0]
        pil = init_state_distn
        nhs = trans_matrix.shape[0]
        giant_Al_pil = np.tile(np.vstack((np.tile(pil, (nhs,1)), Al )), (1,2))
        stateseq = np.empty(T,dtype=np.int32)
        true_segmentation = np.ones(T,dtype=np.int32)
        nextstate_unsmoothed = np.tile(init_state_distn, 2)

        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx] ###check this for the initial and last state and compare with the forward message
            logdomain[nextstate_unsmoothed == 0] = -np.inf

            if np.any(np.isfinite(logdomain)):
                stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            else:
                stateseq[idx] = sample_discrete(nextstate_unsmoothed)
            if stateseq[idx] < nhs: true_segmentation[idx] = 0
            nextstate_unsmoothed = giant_Al_pil[stateseq[idx]]

        return stateseq, true_segmentation


    @property
    def llt(self):
        if self._aBl is None : #or self._aBl is not None:
            data = self.data
            T = self.data.shape[0]
            aBl = self._aBl = np.empty((data.shape[0],self.num_states))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.
            temp_aBl = np.hstack((aBl, aBl))

            nhs = self.num_states
            loglinearterm = np.empty((T, 2 * self.num_states))
            for idx, s in enumerate(self.model.segmentation_distns):
                loglinearterm[:, idx] = self.model.segmentation_distns[idx].log_likelihood(0) #we shouldn't give true samples as we are sampling z and s jointly
                loglinearterm[:, idx + nhs] = self.model.segmentation_distns[idx].log_likelihood(1)
            self._loglinearterm_aBl = temp_aBl + loglinearterm
        return self._loglinearterm_aBl

    @property
    def mf_llt(self):
        if self._mf_aBl is None:
            self.data = np.array(self.data)
            T = self.data.shape[0]
            self._mf_aBl = aBl = np.empty((T,self.num_states))

            for idx, o in enumerate(self.obs_distns):
                aBl[:,idx] = o.expected_log_likelihood(self.data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.
            temp_aBl = np.hstack((aBl, aBl))


            feature_weights = self.feature_weights
            window_data = self.window_data
            nhs = self.num_states
            if self.model.bern_or_weight == 'weight':
                temp_constant = np.sum(feature_weights[:-nhs-1] * window_data[:,:], axis = 1) + feature_weights[-1]
                temp_exp = temp_constant[:, None] + feature_weights[-nhs-1:-1]
                temp_logaddexp = np.logaddexp(0, temp_exp)
                loglinearterm =  np.tile(temp_exp, 2) * np.repeat([0,1], nhs) - np.tile(temp_logaddexp, 2)
            elif self.model.bern_or_weight == 'bern':
                loglinearterm = np.empty((T, 2 * self.num_states))
                for idx, s in enumerate(self.model.segmentation_distns):
                    loglinearterm[:, idx] = self.model.segmentation_distns[idx].expected_log_likelihood()[0]
                    loglinearterm[:, idx + nhs] = self.model.segmentation_distns[idx].expected_log_likelihood()[1]
            self._mf_loglinearterm_aBl = temp_aBl+ loglinearterm
        return self._mf_loglinearterm_aBl



    @property
    def mf_trans_matrix(self):
        return self.model.trans_distn.exp_expected_log_trans_matrix


    @property
    def mf_pi_0(self):
        return self.model.init_state_distn.exp_expected_log_init_state_distn


    def meanfieldupdate(self):
        self.clear_caches()
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_matrix,self.mf_pi_0,self.mf_llt)

    #def _expected_statistics(self,trans_potential,init_potential,likelihood_log_potential, feature_weights):
    def _expected_statistics(self,trans_potential,init_potential, likelihood_log_potential_llt):
        # tic = time.time()
        # alphal = self._messages_forwards_log_slow(trans_potential, init_potential, likelihood_log_potential,
        #                              feature_weights, self.window_data)
        # betal = self._messages_backwards_log_slow(trans_potential, init_potential, likelihood_log_potential,
        #                                           feature_weights, self.window_data)
        # print time.time() - tic
        #
        # tic = time.time()
        alphal = self._messages_forwards_log_fast(trans_potential, init_potential, likelihood_log_potential_llt)
        betal = self._messages_backwards_log_fast(trans_potential, init_potential, likelihood_log_potential_llt)



        # print "fast" + str(time.time() - tic)
        # print ""
        expected_states, normalizer, expected_joints = \
                self._expected_statistics_from_messages(trans_potential,init_potential,
                                                        alphal,betal, likelihood_log_potential_llt)
        expected_transcounts_segmentation = self._expected_transcounts_segmentation(expected_joints)
        expected_initial = self._expected_initial(expected_states, expected_joints)




        assert not np.isinf(expected_states).any()
        return expected_initial, expected_states, expected_transcounts_segmentation, normalizer, expected_joints


    ### EM

    @staticmethod
    def _expected_statistics_from_messages(
            trans_potential, init_potential, alphal, betal, likelihood_log_potential_llt,
            expected_states=None,expected_transcounts=None):

        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:,na]
        np.exp(expected_states,out=expected_states)
        expected_states /= expected_states.sum(1)[:,na]
        sequence_length = betal.shape[0]
        nhs = trans_potential.shape[0]





        #log of initial term:
        # expected segmentation for the initial time point is 1 (1-s_t = 1 or in a tuple of s_t the second element which
        # corresponds to beginning of a new interval is 1
        # log_init_segmentation =  expected_segmentation[:, 1, np.newaxis] * np.log(init_potential)
        temp = np.ones((sequence_length, 2 *nhs, 2 *nhs))#expected_segmentation[:, 0, np.newaxis] #np.ones((sequence_length, nhs)

        pil = np.log(init_potential)

        Al = np.log(trans_potential)


        log_joints = alphal[:-1,:,na] + (betal[1:,na,:] + likelihood_log_potential_llt[1:,na,:]) + \
                     np.tile(np.vstack((np.tile(pil, (nhs,1)), Al )), (1,2))[na,...]

        joints = np.exp(log_joints - np.logaddexp.reduce(alphal[0] + betal[0]))


        normalizer = np.logaddexp.reduce(alphal[0] + betal[0])
        #aprint np.sum(joints) - float(joints.shape[0])
        #assert(np.sum(joints) - float(joints.shape[0]) < 10e-3) #check the posterior transition prob sums to 1
        return expected_states, normalizer, joints #need normalized joints





    @staticmethod
    def _expected_segmentation_states(init_potential, expected_states, trans_potential, expected_joints,
                                      feature_weights, window_data):

        #log_q(s_t) for s_t = 1
        data_length = window_data.shape[0]
        mega_mat = np.hstack((window_data[:data_length - 1,:], expected_states[:data_length - 1,:]))
        temp_1 = np.sum(feature_weights * mega_mat, axis=1)
        with np.errstate(invalid='ignore'):
            temp_2 = np.sum(np.sum(expected_joints[:data_length - 1,:] * np.log(trans_potential), axis = 1), axis = 1)
        log_s_t_1 = temp_1 + temp_2
        log_s_t_1 = np.append(log_s_t_1, -float("inf")) #the last state is always zero so the probability of s_t = 1 is zero

        #log q(s_t) for s_t = 0
        log_s_t_0 = np.sum(expected_states[1:, :] * np.log(init_potential), axis = 1)
        log_s_t_0 = np.append(log_s_t_0, 0)

        temp_stack = np.hstack((log_s_t_1[:, na], log_s_t_0[:, na])) #number of rows is the length of the sequence
        expected_states = np.exp(temp_stack - np.logaddexp.reduce(temp_stack[:,:,na], axis = 1))
        return expected_states

    @staticmethod
    def _expected_initial(expected_states, expected_joints):

        num_hidden_states = expected_joints.shape[1] / 2

        temp_sum = np.transpose(expected_joints[:, :num_hidden_states, :num_hidden_states] +
                                expected_joints[:, :num_hidden_states, num_hidden_states:], (0, 2, 1)).sum((0,2)) # Transposition is to bring the next state to the rows
        #expected_initial = (expected_states[1:, -num_hidden_states:] ).sum(0)
        expected_initial = temp_sum + expected_states[0, :num_hidden_states] + expected_states[0, num_hidden_states:]
        #expected_initial /= expected_initial.sum(0)
        # is normalized
        return expected_initial


    @staticmethod
    def _expected_transcounts_segmentation(expected_joints):
        num_hidden_states = expected_joints.shape[1] / 2
        temp_joints = expected_joints[:,-num_hidden_states:,:num_hidden_states] + expected_joints[:,-num_hidden_states:, num_hidden_states:]
        #don't need the last element in the expected_segmentation; also column zero of expected_segmentation corresponds
        # to prob of s = 1
        expected_transcounts_segmentation = (temp_joints).sum(0) #/temp_joints.sum((1,2))[:,na,na]
        return expected_transcounts_segmentation

    # def log_likelihood(self):
    #     return self._normalizer




class HMMStatesEigen(HMMStatesPython):
    def generate_states(self):
        self.stateseq = sample_markov(
                T=self.T,
                trans_matrix=self.trans_matrix,
                init_state_distn=self.pi_0)

    ### common messages (Gibbs, EM, likelihood calculation)

    @staticmethod
    def _messages_backwards_log(trans_matrix,log_likelihoods):
        from hmm_messages_interface import messages_backwards_log
        return messages_backwards_log(
                trans_matrix,log_likelihoods,
                np.empty_like(log_likelihoods))

    @staticmethod
    def _messages_forwards_log(trans_matrix,init_state_distn,log_likelihoods):
        from hmm_messages_interface import messages_forwards_log
        return messages_forwards_log(trans_matrix,log_likelihoods,
                init_state_distn,np.empty_like(log_likelihoods))

    @staticmethod
    def _messages_forwards_normalized(trans_matrix,init_state_distn,log_likelihoods):
        from hmm_messages_interface import messages_forwards_normalized
        return messages_forwards_normalized(trans_matrix,log_likelihoods,
                init_state_distn,np.empty_like(log_likelihoods))

    # next three methods are just for convenient testing

    def messages_backwards_log_python(self):
        return super(HMMStatesEigen,self)._messages_backwards_log(
                self.trans_matrix,self.aBl)

    def messages_forwards_log_python(self):
        return super(HMMStatesEigen,self)._messages_forwards_log(
                self.trans_matrix,self.pi_0,self.aBl)

    def messages_forwards_normalized_python(self):
        return super(HMMStatesEigen,self)._messages_forwards_normalized(
                self.trans_matrix,self.pi_0,self.aBl)

    ### sampling

    @staticmethod
    def _sample_forwards_log(betal,trans_matrix,init_state_distn,log_likelihoods):
        from hmm_messages_interface import sample_forwards_log
        return sample_forwards_log(trans_matrix,log_likelihoods,
                init_state_distn,betal,np.empty(log_likelihoods.shape[0],dtype='int32'))

    @staticmethod
    def _sample_backwards_normalized(alphan,trans_matrix_transpose):
        from hmm_messages_interface import sample_backwards_normalized
        return sample_backwards_normalized(trans_matrix_transpose,alphan,
                np.empty(alphan.shape[0],dtype='int32'))

    @staticmethod
    def _resample_multiple(states_list):
        from hmm_messages_interface import resample_normalized_multiple
        if len(states_list) > 0:
            loglikes = resample_normalized_multiple(
                    states_list[0].trans_matrix,states_list[0].pi_0,
                    [s.aBl for s in states_list],[s.stateseq for s in states_list])
            for s, loglike in zip(states_list,loglikes):
                s._normalizer = loglike

    ### EM

    @staticmethod
    def _expected_statistics_from_messages(
            trans_potential,likelihood_log_potential,alphal,betal,
            expected_states=None,expected_transcounts=None):
        from hmm_messages_interface import expected_statistics_log
        expected_states = np.zeros_like(alphal) \
                if expected_states is None else expected_states
        expected_transcounts = np.zeros_like(trans_potential) \
                if expected_transcounts is None else expected_transcounts
        return expected_statistics_log(
                np.log(trans_potential),likelihood_log_potential,alphal,betal,
                expected_states,expected_transcounts)

    ### Vitberbi

    def Viterbi(self):
        from hmm_messages_interface import viterbi
        self.stateseq = viterbi(self.trans_matrix,self.aBl,self.pi_0,
                np.empty(self.aBl.shape[0],dtype='int32'))

class HMMStatesEigenSeparateTrans(_SeparateTransMixin,HMMStatesEigen):
    pass

class HMMStatesPossibleChangepoints(_PossibleChangepointsMixin,HMMStatesEigen):
    pass

class HMMStatesPossibleChangepointsSeparateTrans(
        _SeparateTransMixin,
        HMMStatesPossibleChangepoints):
    pass

