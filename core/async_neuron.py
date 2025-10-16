import asyncio
from core.async_neuron_presets import *
import math
import itertools

class AsyncNeuron:
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
    config = DEFAULT_NEURON_CONFIGS[preset.value]
    return cls(**config)

  async def receive_input(self, current_time: float, input_strength: float) -> bool:
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
    dt: float = current_time - self.last_input_time
    return self.rpotential + (self.potential - self.rpotential) * math.exp(-self.decay * dt)

  def _get_dynamic_threshold(self, current_time: float) -> float:
    dt = current_time - self.last_input_time
    if dt < self.arperiod:
      return float('inf')
    elif dt < self.arperiod + self.rrperiod:
      elapsed = dt - self.arperiod
      return self.threshold + (self.threshold * (1 - elapsed / self.rrperiod))
    else:
      return self.threshold

def reset(self, current_time: float = 0) -> None:
    self.potential = self.rpotential
    self.last_input_time = current_time
