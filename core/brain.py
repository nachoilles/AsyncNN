from typing import Optional

from core.neuron import AsyncNeuron

class Brain:
  """A neural network brain that manages neurons and their connections.

  The Brain class serves as a container and manager for a collection of
  AsyncNeuron objects, handling their connections and organization into
  input/output layers.

  Attributes:
    time: Current simulation time
    neurons: Set of all neuron IDs in the brain
    neurons_by_id: Mapping from neuron ID to neuron object
    connections: Mapping from neuron ID to set of connected neuron IDs
    input_neurons: Set of neuron IDs designated as input neurons
    output_neurons: Set of neuron IDs designated as output neurons
  """

  def __init__(self):
    self.time: float = 0.0
    self.neurons: set[int] = set()
    self.neurons_by_id: dict[int, AsyncNeuron] = {}
    self.connections: dict[int, set[int]] = {}
    self.input_neurons: set[int] = set()
    self.output_neurons: set[int] = set()

  def get_neuron(self, id: int) -> Optional[AsyncNeuron]:
    """Retrieve a neuron by its ID.

    Args:
      id: The neuron ID to look up

    Returns:
      The AsyncNeuron object if found, None otherwise
    """
    return self.neurons_by_id.get(id)

  def get_connections(self, id: int) -> set[int]:
    """Get all neurons connected from the specified neuron.

    Args:
      id: Source neuron ID

    Returns:
      Set of target neuron IDs that the source neuron connects to
    """
    return self.connections.get(id, set())

  def add_neuron(self, neuron: AsyncNeuron, is_input: bool = False, is_output: bool = False) -> None:
    """Add a neuron to the brain with optional input/output designation.

    Args:
      neuron: The neuron object to add
      is_input: Whether this neuron is an input neuron
      is_output: Whether this neuron is an output neuron
    """
    if not self.neuron_exists(neuron.id):
      self.neurons.add(neuron.id)
      self.neurons_by_id[neuron.id] = neuron
      self.connections[neuron.id] = set()
      if is_input: self.input_neurons.add(neuron.id)
      if is_output: self.output_neurons.add(neuron.id)

  def delete_neuron(self, neuron_id: int) -> None:
    """Remove a neuron and all its connections from the brain.

    Args:
      neuron_id: ID of the neuron to remove
    """
    if self.neuron_exists(neuron_id):
      self.neurons.remove(neuron_id)
      del self.neurons_by_id[neuron_id]
      connected_neurons = self.connections.get(neuron_id, set())
      del self.connections[neuron_id]
      for connected_neuron in connected_neurons:
        self.connections[connected_neuron].discard(neuron_id)
    self.input_neurons.discard(neuron_id)
    self.output_neurons.discard(neuron_id)

  def connect_neurons(self, from_neuron_id: int, to_neuron_id: int) -> None:
    """Create a directional connection between two neurons.

    Args:
      from_neuron_id: Source neuron ID
      to_neuron_id: Target neuron ID
    """
    if self.neuron_exists(from_neuron_id) and self.neuron_exists(to_neuron_id):
      self.connections[from_neuron_id].add(to_neuron_id)

  def disconnect_neurons(self, from_neuron_id: int, to_neuron_id: int) -> None:
    """Remove a directional connection between two neurons.

    Args:
      from_neuron_id: Source neuron ID
      to_neuron_id: Target neuron ID
    """
    if self.neuron_exists(from_neuron_id):
      self.connections[from_neuron_id].discard(to_neuron_id)

  def sever_connection(self, neuron_a_id: int, neuron_b_id: int) -> None:
    """Remove bidirectional connections between two neurons.

    Args:
      neuron_a_id: First neuron ID
      neuron_b_id: Second neuron ID
    """
    if self.neuron_exists(neuron_a_id) and self.neuron_exists(neuron_b_id):
      self.connections[neuron_a_id].discard(neuron_b_id)
      self.connections[neuron_b_id].discard(neuron_a_id)

  def neuron_exists(self, id: int) -> bool:
    """Check if a neuron exists in the brain.

    Args:
      id: Neuron ID to check

    Returns:
      True if neuron exists, False otherwise
    """
    return id in self.neurons