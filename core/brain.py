import asyncio
from typing import Optional
from dataclasses import dataclass

from core.async_neuron import AsyncNeuron
from core.synapse import Synapse, SynapseMap

@dataclass
class Input:
  neuron_id: int    # ID of the neuron receiving the input signal
  strength: float   # Strength of the signal

class Brain:
  def __init__(self):
    self.time: float = 0.0
    self._neurons: set[int] = set()
    self._neurons_by_id: dict[int, AsyncNeuron] = {}
    self._synapse_map: SynapseMap = SynapseMap()
    self._input_buffer: dict[float, list[Input]] = {}

  # ---------- Getters and helper methods ----------

  @property
  def neurons(self) -> set[int]:
    return self._neurons

  @property
  def neurons_by_id(self) -> dict[int, AsyncNeuron]:
    return self._neurons_by_id

  @property
  def synapse_map(self) -> SynapseMap:
    return self._synapse_map

  @property
  def input_buffer(self) -> dict[float, list[Input]]:
    return self._input_buffer

  def get_neuron(self, id: int) -> Optional[AsyncNeuron]:
    return self._neurons_by_id.get(id)

  def get_pre_synapses(self, neuron_id: int) -> list[Synapse]:
    if neuron_id in self._synapse_map.pre_synapses:
      pre_synapse_ids = self._synapse_map.pre_synapses[neuron_id]
      return [self._synapse_map.synapses[pre_id] for pre_id in pre_synapse_ids]
    else:
      return []

  def get_post_synapses(self, neuron_id: int) -> list[Synapse]:
    if neuron_id in self._synapse_map.post_synapses:
      post_synapse_ids = self._synapse_map.post_synapses[neuron_id]
      return [self._synapse_map.synapses[post_id] for post_id in post_synapse_ids]
    else:
      return []

  # ---------- Neuron management ----------

  def add_neuron(self, neuron: AsyncNeuron) -> None:
    if neuron.id not in self._neurons:
      self._neurons.add(neuron.id)
      self._neurons_by_id[neuron.id] = neuron
      self._synapse_map.add_neuron(neuron.id)

  def delete_neuron(self, neuron: int | AsyncNeuron) -> None:
    if isinstance(neuron, int):
      neuron_id = neuron
    else:
      neuron_id = neuron.id

    if neuron_id in self._neurons:
      self.synapse_map.disconnect_neuron(neuron_id)
      for timestamp in self._input_buffer:
        self._input_buffer[timestamp] = [
          input for input in self._input_buffer[timestamp]
          if input.neuron_id != neuron_id
        ]
      if neuron_id in self._neurons_by_id:
        del self._neurons_by_id[neuron_id]
      self._neurons.remove(neuron_id)

  # ---------- Input handling ----------

  def add_input(self, timestamp: float, input: Input) -> None:
    self._input_buffer.setdefault(timestamp, []).append(input)

  # ---------- Propagation loop ----------

  async def propagate(self) -> None:
    if not self._input_buffer:
      return

    t_next = min(self._input_buffer.keys())
    self.time = t_next

    events = self._input_buffer.pop(self.time, [])
    tasks = [
      self._process_input(inp.neuron_id, inp.strength) for inp in events
    ]

    if tasks:
      await asyncio.gather(*tasks)

  async def _process_input(self, neuron_id: int, strength: float) -> None:
    neuron = self._neurons_by_id[neuron_id]
    fired = await neuron.receive_input(self.time, strength)
    if fired:
      await self._propagate_from(neuron.id)

  async def _propagate_from(self, neuron_id: int) -> None:
    for syn_id in self.synapse_map.pre_synapses[neuron_id]:
      syn = self.synapse_map.synapses[syn_id]
      arrival_time = self.time + syn.delay
      input_signal = Input(syn.post_id, syn.weight)
      self.add_input(arrival_time, input_signal)