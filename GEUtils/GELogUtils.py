# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 08:35:58 2019

@author: siddh
"""

import json
from texttable import Texttable


def log_setup(args_in):
    """
    Function to setup the logging hash table.
    """    
    log = dict()
    log["times"] = []
    log["losses"] = []
    log["cluster_quality"] = []
    log["params"] = vars(args_in)
    return log
