import asyncio
import random
from typing import Optional
from dataclasses import dataclass

from core.async_neuron import AsyncNeuron

@dataclass
class Synapse:
  """Represents a directional connection between two neurons.

  Attributes:
    target_id: ID of the neuron this synapse connects to
    weight: Strength of the synapse
    delay: Time delay for the signal to propagate
  """
  target_id: int
  weight: float
  delay: float

  @staticmethod
  def default_weight_for(
    neuron: "AsyncNeuron",
    fraction: float = 0.5) -> float:
    """
    Suggest a reasonable EPSP/IPSP weight for a postsynaptic neuron.

    Computes a weight as a fraction of the neuron's threshold minus resting
    potential, ensuring a single spike has a reasonable postsynaptic effect.


    Args:
      neuron (AsyncNeuron): The postsynaptic neuron to target.
      fraction (float): Fraction of (threshold - resting potential) to use.
        Positive fractions correspond to excitatory inputs (EPSPs).
        Negative fractions correspond to inhibitory inputs (IPSPs).

    Returns:
      float: Suggested synapse weight for spike-driven input.
    """
    return fraction * (neuron.threshold - neuron.rpotential)

  @staticmethod
  def delay_from_distr(mean: float = 0.002, std: float = 0.0005) -> float:
    """
    Computes a synaptic delay in seconds for a local cortical connection
    from a gaussian (normal) distribution.

    Args:
      mean (float): Mean delay in seconds (default 0.002 s ~2 ms)
      std (float): Standard deviation for variability (default 0.0005 s
      ~0.5 ms)

    Returns:
      float: Suggested synaptic delay
    """
    return max(0.0, random.gauss(mean, std))

@dataclass
class Input:
  """Represents an input event for a neuron at a specific time.

  Attributes:
    neuron_id: ID of the neuron receiving the input
    strength: Magnitude of the input
  """
  neuron_id: int
  strength: float

class Brain:
  """A spiking neural network brain simulation.

  Manages neurons, their connections, and propagates signals asynchronously
  through the network.

  Attributes:
    time: Current simulation time
    frequency: Simulation frequency (Hz)
    dt: Time step derived from frequency
    neurons: Set of all neuron IDs
    neurons_by_id: Mapping from neuron ID to AsyncNeuron object
    connections: Mapping from neuron ID to list of Synapse objects
    input_neurons: Set of neuron IDs designated as inputs
    output_neurons: Set of neuron IDs designated as outputs
    input_buffer: Mapping of timestamps to lists of Input events
  """

  def __init__(self):
    """Initialize a Brain instance.

    Args:
      frequency: Simulation frequency in Hz (default 10.0)
    """
    self.time: float = 0.0
    self.neurons: set[int] = set()
    self.neurons_by_id: dict[int, AsyncNeuron] = {}
    self.connections: dict[int, list[Synapse]] = {}
    self.input_neurons: set[int] = set()
    self.output_neurons: set[int] = set()
    self.input_buffer: dict[float, list[Input]] = {}

  # ---------- Getters and helper methods ----------

  def get_neuron(self, id: int) -> Optional[AsyncNeuron]:
    """Retrieve a neuron by its ID.

    Args:
      id: Neuron ID to look up

    Returns:
      AsyncNeuron object if found, None otherwise
    """
    return self.neurons_by_id.get(id)

  def get_synapses(self, neuron_id: int) -> list[Synapse]:
    """
    Retrieve connections from the brain.

    Args:
      neuron_id: neuron ID to get its connections

    Returns:
      A dictionary mapping neuron IDs to lists of Synapse objects.
      If neuron_id is given, returns a dictionary with only that entry (if
      it exists).
    """
    if neuron_id in self.connections:
      return self.connections[neuron_id]
    else:
      return []

  def neuron_exists(self, neuron_id: int) -> bool:
    """Check if a neuron exists in the brain.

    Args:
      neuron_id: Neuron ID

    Returns:
      True if neuron exists, False otherwise
    """
    return neuron_id in self.neurons

  def is_input_neuron(self, neuron_id: int) -> bool:
    """Check if a give neuron is an input neuron.

    Args:
      neuron_id: Neuron ID

    Returns:
      True if neuron is an input neuron, False otherwise
    """
    return neuron_id in self.input_neurons

  def is_output_neuron(self, neuron_id: int) -> bool:
    """Check if a give neuron is an output neuron.

    Args:
      neuron_id: Neuron ID

    Returns:
      True if neuron is an output neuron, False otherwise
    """
    return neuron_id in self.output_neurons

  # ---------- Neuron management ----------

  def add_neuron(
      self,
      neuron: AsyncNeuron,
      is_input: bool = False,
      is_output: bool = False
    ) -> None:
    """Add a neuron to the brain with optional input/output designation.

    Args:
      neuron: AsyncNeuron instance to add
      is_input: Mark neuron as input (default False)
      is_output: Mark neuron as output (default False)
    """
    if not self.neuron_exists(neuron.id):
      self.neurons.add(neuron.id)
      self.neurons_by_id[neuron.id] = neuron
      self.connections[neuron.id] = []
      if is_input:
        self.input_neurons.add(neuron.id)
      if is_output:
        self.output_neurons.add(neuron.id)

  def delete_neuron(self, neuron_id: int) -> None:
    """Completely remove a neuron and all its associated data.

    This method deletes the specified neuron from the brain, including:
    - Its registration in the neuron sets (`neurons`, `input_neurons`,
      `output_neurons`)
    - Its connections in `self.connections`
    - Any queued inputs referencing it in `self.input_buffer`
    - Its entry in `self.neurons_by_id`

    Args:
      neuron_id: The ID of the neuron to delete.
    """
    if self.neuron_exists(neuron_id):
      if self.is_input_neuron(neuron_id):
        self.input_neurons.remove(neuron_id)
      if self.is_output_neuron(neuron_id):
        self.output_neurons.remove(neuron_id)
      self.disconnect_neuron(neuron_id)
      for timestamp in self.input_buffer:
        self.input_buffer[timestamp] = [
          input for input in self.input_buffer[timestamp]
          if input.neuron_id != neuron_id
        ]
      if neuron_id in self.neurons_by_id:
        del self.neurons_by_id[neuron_id]
      if neuron_id in self.neurons:
        self.neurons.remove(neuron_id)

  def create_synapse(self, from_neuron_id: int, synapse: Synapse) -> None:
    """Connect one neuron to another with a synapse.

    Args:
      from_neuron_id: Source neuron ID
      synapse: Synapse object specifying target, weight, and delay
    """
    if self.neuron_exists(from_neuron_id) \
      and self.neuron_exists(synapse.target_id):
      self.connections[from_neuron_id].append(synapse)

  def break_synapses(self, from_neuron_id: int, to_neuron_id: int) -> None:
    """Remove all synapses from one neuron to another.

    Args:
      from_neuron_id: Source neuron ID
      to_neuron_id: Target neuron ID
    """
    if self.neuron_exists(from_neuron_id):
      self.connections[from_neuron_id] = [
        syn for syn in self.connections[from_neuron_id]
        if syn.target_id != to_neuron_id
      ]

  def sever_connection(self, neuron_a_id: int, neuron_b_id: int) -> None:
    """Remove all bidirectional connections between two neurons.

    Args:
      neuron_a_id: First neuron ID
      neuron_b_id: Second neuron ID
    """
    if self.neuron_exists(neuron_a_id) and self.neuron_exists(neuron_b_id):
      self.break_synapses(neuron_a_id, neuron_b_id)
      self.break_synapses(neuron_b_id, neuron_a_id)

  def disconnect_neuron(self, neuron_id: int) -> None:
    if neuron_id in self.connections:
      del self.connections[neuron_id]
    for other_id in self.connections:
      self.connections[other_id] = [
        syn for syn in self.connections[other_id] if syn.target_id != neuron_id
      ]

  # ---------- Input handling ----------

  def add_input(self, timestamp: float, input: Input) -> None:
    """Schedule an input event for a neuron at a specific time.

    Args:
      timestamp: Simulation time at which input occurs
      input: Input object containing neuron ID and strength
    """
    self.input_buffer.setdefault(timestamp, []).append(input)

  # ---------- Propagation loop ----------

  async def propagate(self) -> None:
    """Process all input events scheduled for the next simulation time.

    Advances the simulation to the next event timestamp, applies inputs
    to neurons, and propagates any resulting firings.
    """
    if not self.input_buffer:
      return

    t_next = min(self.input_buffer.keys())
    self.time = t_next

    events = self.input_buffer.pop(self.time, [])
    tasks = [
      self._process_input(inp.neuron_id, inp.strength) for inp in events
    ]

    if tasks:
      await asyncio.gather(*tasks)

  async def _process_input(self, neuron_id: int, strength: float) -> None:
    """Apply input to a neuron and propagate if it fires.

    Args:
      neuron_id: ID of the neuron receiving input
      strength: Magnitude of the input
    """
    neuron = self.neurons_by_id[neuron_id]
    fired = await neuron.receive_input(self.time, strength)
    if fired:
      await self._propagate_from(neuron.id)

  async def _propagate_from(self, neuron_id: int) -> None:
    """Propagate a neuron's firing to connected neurons via synapses.

    Args:
      neuron_id: ID of the neuron that fired
    """
    for syn in self.connections.get(neuron_id, []):
      arrival_time = self.time + syn.delay
      input_signal = Input(syn.target_id, syn.weight)
      self.add_input(arrival_time, input_signal)