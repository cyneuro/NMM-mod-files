import glob
import json
import os
import numpy as np
from neuron import h
from bmtk.simulator.bionet.pyfunction_cache import add_synapse_model

rng = np.random.default_rng(0)

TEST_NUM = 20
TEST_COUNT1 = 0
TEST_COUNT2 = 0

LOGN_PARAM_DICT = {}


def get_logn_params(m, s, sigma_lower, sigma_upper):
    key = (m, s)
    params = LOGN_PARAM_DICT.get(key)
    if params is None:
        if m:
            sigma2 = np.log((s / m) ** 2 + 1)
            params = [np.log(m) - sigma2 / 2, sigma2 ** 0.5, {}]
        else:
            params = [-np.inf, 0., {}]
        LOGN_PARAM_DICT[key] = params
    key = (sigma_lower, sigma_upper)
    bounds = params[2].get(key)
    if bounds is None:
        mu, sigma = params[:2]
        lo = None if sigma_lower is None else np.exp(mu + sigma_lower * sigma)
        up = None if sigma_upper is None else np.exp(mu + sigma_upper * sigma)
        bounds = (lo, up)
        params[2][key] = bounds
    return params[0], params[1], bounds


def gen_logn_weight(mean, stdev, sigma_lower=None, sigma_upper=None):
    mu, sigma, bounds = get_logn_params(mean, stdev, sigma_lower, sigma_upper)
    weight = rng.lognormal(mu, sigma)
    if bounds[0] is not None:
        weight = max(weight, bounds[0])
    if bounds[1] is not None:
        weight = min(weight, bounds[1])
    return weight


def get_edge_prop_func(edge_props):
    """Get value from Edge object. Return None if property does not exist."""
    sonata_edge = edge_props._edge

    def get_edge_prop(key, default=None):
        return sonata_edge[key] if key in sonata_edge else default
    return get_edge_prop


def lognormal_weight(edge_props, source, target):
    """Function for synaptic weight between nodes"""
    get_edge_prop = get_edge_prop_func(edge_props)
    mean = get_edge_prop('syn_weight')
    if mean is None:
        weight = 1.0
    else:
        stdev = get_edge_prop('weight_sigma')
        if stdev:
            weight = gen_logn_weight(
                mean, stdev,
                sigma_lower=get_edge_prop('sigma_lower_bound'),
                sigma_upper=get_edge_prop('sigma_upper_bound'))
        else:
            weight = mean
        # global TEST_COUNT1
        # if TEST_COUNT1 < TEST_NUM:
        #     print(f'Synapse weight: {weight: .4g}')
        #     TEST_COUNT1 += 1
    return weight


def set_syn_weight(syn, syn_params):
    """Change initW property in synapse point process.
    Alternative method to change synaptic weight."""
    initW = syn_params.get('initW')
    stdevW = syn_params.get('stdevW')
    if initW is not None:
        if stdevW:
            initW = gen_logn_weight(
                initW, stdevW,
                sigma_lower=syn_params.get('sigma_lower_bound'),
                sigma_upper=syn_params.get('sigma_upper_bound'))
        syn.initW = initW
        # global TEST_COUNT2
        # if TEST_COUNT2 < TEST_NUM:
        #     print(f'Synapse initW: {initW: .4g}')
        #     TEST_COUNT2 += 1


Exp2Syn_STP_params = ('e', 'tau1', 'tau2', 'Use', 'Dep', 'Fac')
EXP2SYN_depressing_params = ("d1", "d2", "TauD1","TauD2","TauF","F",'initW')

def Exp2Syn_STP(syn_params, sec_x, sec_id):
    """Create a Exp2Syn_STP synapse
    :param syn_params: parameters of a synapse
    :param sec_x: normalized distance along the section
    :param sec_id: target section
    :return: NEURON synapse object
    """
    syn = h.Exp2Syn_STP(sec_x, sec=sec_id)
    for key in Exp2Syn_STP_params:
        value = syn_params.get(key)
        if value is not None:
            setattr(syn, key, value)
    set_syn_weight(syn, syn_params)
    return syn


def exp2syn_stp(syn_params, xs, secs):
    """Create a list of Exp2Syn_STP synapses
    :param syn_params: parameters of a synapse
    :param xs: list of normalized distances along the section
    :param secs: target sections
    :return: list of NEURON synpase objects
    """
    return [Exp2Syn_STP(syn_params, x, sec) for x, sec in zip(xs, secs)]


def Exp2Syn1_STSP(syn_params, sec_x, sec_id):
    """Create a Exp2Syn_STP synapse
    :param syn_params: parameters of a synapse
    :param sec_x: normalized distance along the section
    :param sec_id: target section
    :return: NEURON synapse object
    """
    syn = h.Exp2Syn1_STSP(sec_x, sec=sec_id)
    for key in Exp2Syn_STP_params:
        value = syn_params.get(key)
        if value is not None:
            setattr(syn, key, value)
    set_syn_weight(syn, syn_params)
    return syn


def exp2syn1_STSP(syn_params, xs, secs):
    """Create a list of Exp2Syn_STP synapses
    :param syn_params: parameters of a synapse
    :param xs: list of normalized distances along the section
    :param secs: target sections
    :return: list of NEURON synpase objects
    """
    return [Exp2Syn_STP(syn_params, x, sec) for x, sec in zip(xs, secs)]


def load(randseed=1111, rng_obj=None):
    global rng
    if rng_obj is None:
        rng = np.random.default_rng(randseed)
    else:
        rng = rng_obj
    add_synapse_model(Exp2Syn_STP, 'Exp2Syn_STP', overwrite=False)
    add_synapse_model(Exp2Syn_STP, overwrite=False)

    add_synapse_model(Exp2Syn1_STSP, 'Exp2Syn1_STSP',overwrite=False)
    add_synapse_model(Exp2Syn1_STSP, overwrite=False)


def syn_params_dicts(syn_dir='components/synaptic_models'):
    """
    returns: A dictionary of dictionaries containing all
    properties in the synapse json files
    """
    files = glob.glob(os.path.join(syn_dir, '*.json'))
    data = {}
    for fh in files:
        with open(fh) as f:
            # data["filename.json"] = {"prop1":"val1", ...}
            data[os.path.basename(fh)] = json.load(f)
    return data
