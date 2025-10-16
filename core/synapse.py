import itertools
from typing import Optional
import random

class Synapse:
  _get_id = itertools.count()

  def __init__(self, pre_id: int, post_id: int, weight: float, delay: float) -> None:
    self._id: int = next(self._get_id)        # Id for the actual synapse
    self.pre_id = pre_id                     # Id of the pre-synaptic neuron
    self.post_id = post_id                   # Id of the post-synaptic neuron
    self.weight = weight                     # Weight of the synapse
    self.delay = delay                       # Delay of the synapse

  @property
  def id(self) -> int:
    return self._id

  @staticmethod
  def get_fractional_weight(threshold: float, rest_potential: float , fraction: float = 0.5) -> float:
    return fraction * (threshold - rest_potential)

  @staticmethod
  def get_delay_from_distr(mean: float = 0.002, std: float = 0.0005) -> float:
    return max(0.0, random.gauss(mean, std))

class SynapseMap:
  def __init__(self) -> None:
    self._synapses: dict[int, Synapse] = {}              # Set of all synapses
    self._pre_synapses: dict[int, list[int]] = {}     # [pre-neuron-id, [pre-synapse-ids]]
    self._post_synapses: dict[int, list[int]] = {}    # [post-neuron-id, [post-synapse-ids]]

  @property
  def synapses(self) -> dict[int, Synapse]:
    return self._synapses

  @property
  def pre_synapses(self) -> dict[int, list[int]]:
    return self._pre_synapses

  @property
  def post_synapses(self) -> dict[int, list[int]]:
    return self._post_synapses

  def create_synapse(self, synapse: Synapse) -> None:
    if synapse.pre_id is not None and synapse.post_id is not None:
      self._synapses[synapse.id] = synapse
      self._pre_synapses.setdefault(synapse.pre_id, []).append(synapse.id)
      self._post_synapses.setdefault(synapse.post_id, []).append(synapse.id)


  def break_synapse(self, syn_id: int) -> None:
    del self._synapses[syn_id]

    for pre_id, syn_list in list(self._pre_synapses.items()):
      if syn_id in syn_list:
        syn_list.remove(syn_id)

    for post_id, syn_list in list(self._post_synapses.items()):
      if syn_id in syn_list:
        syn_list.remove(syn_id)

  def disconnect_neuron(self, neuron_id: int, delete: bool = False) -> None:
    pre_syns = self._pre_synapses[neuron_id]
    post_syns = self._post_synapses[neuron_id]
    for syn in pre_syns + post_syns:
      if syn in self._synapses:
        del self._synapses[syn]

    if neuron_id in self._pre_synapses:
      if delete:
        del self._pre_synapses[neuron_id]
      else:
        self._pre_synapses[neuron_id] = []

    if neuron_id in self._post_synapses:
      if delete:
        del self._pre_synapses[neuron_id]
      else:
        self._post_synapses[neuron_id] = []

  def add_neuron(self, neuron_id: int) -> None:
    if not self._pre_synapses[neuron_id]:
      self._pre_synapses[neuron_id] = []
    if not self._post_synapses[neuron_id]:
      self._post_synapses[neuron_id] = []