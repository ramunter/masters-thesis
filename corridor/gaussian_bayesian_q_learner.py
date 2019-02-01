from numpy.random import binomial
from numpy import array
import numpy as np

from collections import namedtuple

from util import featurizer


MeanParameters = namedtuple('Mean Parameters', ['state_mean', 
                                       'action_mean',
                                       'const_mean',

SDParameters = namedtuple('SD Parameters', ['state_sd',
                                            'action_sd',
                                            'const_sd'])



