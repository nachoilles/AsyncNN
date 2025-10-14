from enum import Enum

DEFAULT_NEURON_CONFIGS: dict[str, dict] = {
  "biological": {
    "rpotential": -70.0,    # typical resting potential in mV
    "threshold": -55.0,     # typical firing threshold
    "decay": 0.05,          # realistic membrane time constant (scaled for simulation units)
    "psdelay": 0.001,       # synaptic delay in seconds
    "arperiod": 0.002,      # absolute refractory period in seconds
    "rrperiod": 0.003,      # relative refractory period in seconds
  },
  "non_refractory": {
    "rpotential": -70.0,
    "threshold": -55.0,
    "decay": 0.05,
    "psdelay": 0.001,
    "arperiod": 0.0,        # no absolute refractory period
    "rrperiod": 0.0,        # no relative refractory period
  },
  "fast_spiking": {
    "rpotential": -65.0,
    "threshold": -50.0,
    "decay": 0.1,
    "psdelay": 0.0005,
    "arperiod": 0.001,
    "rrperiod": 0.002,
  },
  "high_threshold": {
    "rpotential": -70.0,
    "threshold": -40.0,
    "decay": 0.05,
    "psdelay": 0.001,
    "arperiod": 0.002,
    "rrperiod": 0.003,
  },
}

class Preset(Enum):
  BIOLOGICAL = "biological"
  NON_REFRACTORY = "non_refractory"
  FAST_SPIKING = "fast_spiking"
  HIGH_THRESHOLD = "high_threshold"