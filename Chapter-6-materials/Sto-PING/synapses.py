from neuron import h
from bmtk.simulator.bionet.pyfunction_cache import add_synapse_model
import glob
import os
import json

def exp2syn_weight(syn_params, sec_x, sec_id):

    lsyn = h.Exp2Syn_weight(sec_x, sec=sec_id)

    if syn_params.get('initW'):
        lsyn.initW = float(syn_params['initW'])

    if syn_params.get('tau1'):
        lsyn.tau1 = float(syn_params['tau1'])

    if syn_params.get('tau2'):
        lsyn.tau2 = float(syn_params['tau2'])

    if syn_params.get('erev'):
        lsyn.erev = float(syn_params['erev'])

    return lsyn

def load():
    add_synapse_model(exp2syn_weight, 'exp2syn_weight', overwrite=False)
    add_synapse_model(exp2syn_weight, overwrite=False)

def syn_params_dicts(syn_dir='components/synaptic_models'):
    """
    returns: A dictionary of dictionaries containing all
    properties in the synapse json files
    """
    files = glob.glob(os.path.join(syn_dir,'*.json'))
    data = {}
    for fh in files:
        with open(fh) as f:
            data[os.path.basename(fh)] = json.load(f) #data["filename.json"] = {"prop1":"val1",...}
    return data