import asyncio
from core.async_neuron_presets import *
import math
import itertools

class AsyncNeuron:
  """
  Asynchronous leaky integrate-and-fire neuron model.

  This neuron integrates incoming inputs (EPSPs or IPSPs) asynchronously,
  with an exponentially decaying membrane potential toward a resting potential.
  It fires an action potential when the potential exceeds a threshold, and
  models both absolute and relative refractory periods.

  Attributes:
    id (int): Unique identifier for the neuron instance.
    potential (float): Current membrane potential.
    rpotential (float): Resting membrane potential toward which the neuron decays.
    threshold (float): Potential level required to trigger an action potential.
    decay (float): Rate constant for exponential decay toward resting potential.
    psdelay (float): Post-synaptic delay before input affects the neuron.
    arperiod (float): Absolute refractory period during which no spike can occur.
    rrperiod (float): Relative refractory period during which threshold is elevated.
    last_input_time (float): Simulation time of the most recent input received.
  """
  _id_gen = itertools.count()

  def __init__(
    self,
    rpotential: float = 0.0,
    threshold: float = 1.0,
    decay: float = 0.9,
    psdelay: float = 0.0,
    arperiod: float = 0.0,
    rrperiod: float = 0.0
  ):
    """
    Initialize a new AsyncNeuron instance.

    Args:
      rpotential (float): Resting membrane potential (default 0.0).
      threshold (float): Firing threshold potential (default 1.0).
      decay (float): Membrane potential decay rate per time unit (default 0.9).
      psdelay (float): Post-synaptic delay before input affects potential (default 0.0).
      arperiod (float): Absolute refractory period duration (default 0.0).
      rrperiod (float): Relative refractory period duration (default 0.0).
    """
    self.id: int = next(self._id_gen)
    self.potential: float = rpotential
    self.rpotential: float = rpotential
    self.threshold: float = threshold
    self.decay: float = decay
    self.psdelay: float = psdelay
    self.arperiod: float = arperiod
    self.rrperiod: float = rrperiod
    self.last_input_time: float = 0.0

  @classmethod
  def from_preset(cls, preset: Preset) -> "AsyncNeuron":
    """
    Create an AsyncNeuron instance using a predefined preset.

    Args:
      preset (Preset): One of the predefined presets.

    Returns:
      AsyncNeuron: A new neuron instance with parameters from the preset.
    """
    config = DEFAULT_NEURON_CONFIGS[preset.value]
    return cls(**config)

  async def receive_input(self, current_time: float, input_strength: float) -> bool:
    """
    Process an incoming signal for the neuron asynchronously.

    The membrane potential decays toward the resting potential based on time
    since the last input, then the input strength is applied. If the resulting
    potential exceeds the current dynamic threshold, the neuron fires and
    the potential is reset to resting potential.

    Args:
      current_time (float): The current simulation time.
      input_strength (float): Strength of the incoming signal (positive for EPSP, negative for IPSP).

    Returns:
      bool: True if the neuron fires after this input, False otherwise.
    """
    await asyncio.sleep(self.psdelay)
    old_potential = self.get_current_potential(current_time)
    self.potential = old_potential
    self.last_input_time = current_time

    self.potential += input_strength

    threshold_now = self._get_dynamic_threshold(current_time)
    fired = self.potential >= threshold_now
    if fired:
        self.potential = self.rpotential
    return fired

  def get_current_potential(self, current_time: float) -> float:
    """
    Calculate the current membrane potential accounting for exponential decay.

    The potential decays toward the resting potential according to the
    elapsed time since the last input.

    Args:
      current_time (float): The simulation time for which to calculate potential.

    Returns:
      float: The decayed membrane potential at the specified time.
    """
    dt: float = current_time - self.last_input_time
    return self.rpotential + (self.potential - self.rpotential) * math.exp(-self.decay * dt)

  def _get_dynamic_threshold(self, current_time: float) -> float:
    """
    Compute the neuron's dynamic threshold considering refractory periods.

    During the absolute refractory period, firing is impossible.
    During the relative refractory period, the threshold is elevated.
    After the relative refractory period, the threshold returns to baseline.

    Args:
      current_time (float): The current simulation time.

    Returns:
      float: The effective threshold at the given time.
    """
    dt = current_time - self.last_input_time
    if dt < self.arperiod:
      return float('inf')
    elif dt < self.arperiod + self.rrperiod:
      elapsed = dt - self.arperiod
      # Linearly decreasing threshold during relative refractory period
      return self.threshold + (self.threshold * (1 - elapsed / self.rrperiod))
    else:
      return self.threshold

def reset(self, current_time: float = 0) -> None:
    """
    Reset the neuron to its resting potential and set last input time.

    Args:
      current_time (float): Time to set as last input time (default 0).
    """
    self.potential = self.rpotential
    self.last_input_time = current_time

def __repr__(self) -> str:
    """
    Return a string representation of the neuron's fixed properties.

    Returns:
      str: String describing the neuron ID, threshold, and decay rate.
    """
    return f"AsyncNeuron(id={self.id}, threshold={self.threshold}, decay={self.decay})"