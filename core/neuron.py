import asyncio
import math
import itertools

class AsyncNeuron:
  """A neuron model that processes inputs asynchronously with potential decay.

  Each neuron maintains an internal potential that decays over time and fires
  when the potential reaches a specified threshold. The model follows a
  leaky integrate-and-fire mechanism with reset after firing.

  Attributes:
    id: Unique identifier for the neuron instance
    potential: Current membrane potential (resets to 0 after firing)
    threshold: Potential level required for the neuron to fire
    decay: Rate at which potential decreases per time unit
    last_input_time: Timestamp of the most recent input reception
  """
  _id_gen = itertools.count()

  def __init__(self, threshold: float = 1.0, decay: float = 0.9, delay: float = 0.0):
    """Initialize a new neuron with specified parameters.

    Args:
      threshold: Firing threshold potential (default 1.0)
      decay: Decay rate per time unit (default 0.9)
      delay: Delay from neuron receiving an input to updating it state (default 0.0)
    """
    self.id: int = next(self._id_gen)
    self.potential: float = 0.0
    self.threshold: float = threshold
    self.decay: float = decay
    self.delay: float = delay
    self.last_input_time: float = 0.0

  async def receive_input(self, current_time: float, input_strength: float) -> bool:
    """
    Process an incoming signal for the neuron at the given simulation time.

    This coroutine updates the neuron's membrane potential based on the
    provided input strength and determines whether the neuron fires as a
    result. If the firing threshold is reached, the potential resets to zero.

    Args:
      current_time: The current simulation time (in seconds).
      input_strength: The strength of the incoming signal.

    Returns:
      True if the neuron fires after processing the input, False otherwise.
    """
    await asyncio.sleep(self.delay)
    old_potential = self.get_current_potential(current_time)
    self.potential = old_potential
    self.last_input_time = current_time
    self.potential += input_strength
    fired = self.potential >= self.threshold
    if fired:
      self.potential = 0.0
    return fired


  def get_current_potential(self, current_time: float) -> float:
    """Calculate the current potential accounting for time decay.

    Computes what the potential would be at the given time by applying
    decay since the last input, without modifying the neuron's state.

    Args:
      current_time: Time to calculate potential for

    Returns:
      The decayed potential value at the specified time
    """
    delta_time: float = current_time - self.last_input_time
    return max(self.potential * math.exp(-self.decay * delta_time), 0.0)

  def reset(self, current_time: float = 0) -> None:
    """Reset the neuron to its initial state.

    Args:
      current_time: Time to set as the last input time (default 0)
    """
    self.potential = 0.0
    self.last_input_time = current_time

  def __repr__(self) -> str:
    """Return a string representation of the neuron's fixed properties."""
    return f"AsyncNeuron(id={self.id}, threshold={self.threshold}, decay={self.decay})"