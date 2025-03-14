from __future__ import annotations

from dataclasses import dataclass, field

ComponentIndex = int


@dataclass
class Component:
    """Structure that represents a segmentation unit in the algorithm."""

    nodes: set[tuple[int, int]]
    component_indexes: set[ComponentIndex]
    ind: float = field(default=0)

    def __len__(self) -> int:
        """Magic method for finding size of a component.

        Returns
        -------
        int
            Size of a component.
        """
        return len(self.nodes)

    def merge(self, other: Component, ind: float) -> None:
        """Merge other component into this one.

        Parameters
        ----------
        other : Component
            Componned that will be merged.
        ind : float
            The weight of an edge this component will be merged on.
        """
        self.nodes = self.nodes.union(other.nodes)
        self.component_indexes = self.component_indexes.union(other.component_indexes)
        self.ind = ind
