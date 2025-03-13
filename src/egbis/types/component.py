from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain

import networkx as nx
import numpy as np

Edge = tuple[tuple[int, int], tuple[int, int]]
Directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def intersect_edges(edges_a: set[Edge], edges_b: set[Edge]) -> set[Edge]:
    """Find edges that are both in edges_a set and edges_b set.

    Parameters
    ----------
    edges_a : set[Edge]
        Sets of edges.
    edges_b : set[Edge]
        Sets of edges.

    Returns
    -------
    set[Edge]
        Intersection of both provided sets of edges.
    """
    result = set()
    _union = edges_a.union(edges_b)

    for edge in _union:

        if (edge not in edges_a) and (edge[::-1] not in edges_a):
            continue

        if (edge not in edges_b) and (edge[::-1] not in edges_b):
            continue

        if edge[::-1] in result:
            continue

        result.add(edge)

    return result


def union_edges(edges_a: set[Edge], edges_b: set[Edge]) -> set[Edge]:
    """Find edges that are either in edges_a set or in edges_b set.

    Parameters
    ----------
    edges_a : set[Edge]
        Sets of edges.
    edges_b : set[Edge]
        Sets of edges.

    Returns
    -------
    set[Edge]
        Union of both provided sets of edges.
    """
    result = set()
    _union = edges_a.union(edges_b)

    for edge in _union:

        if edge[::-1] in result:
            continue

        result.add(edge)

    return result


def subtract_edges(edges_a: set[Edge], edges_b: set[Edge]) -> set[Edge]:
    """Find edges that are among edges_a, but not among edges_b.

    Parameters
    ----------
    edges_a : set[Edge]
        Sets of edges.
    edges_b : set[Edge]
        Sets of edges.

    Returns
    -------
    set[Edge]
        Edges that are among edges_a, but not among edges_b.
    """
    result = set()

    for edge in edges_a:

        if edge in edges_b or edge[::-1] in edges_b:
            continue

        result.add(edge)

    return result


@dataclass
class Component:
    """Structure that represents a segmentation unit in the algorithm."""

    nodes: set[tuple[int, int]]
    external_edges: set[Edge]
    partition: np.ndarray
    edges: set[Edge] = field(default_factory=lambda: set())

    def __len__(self) -> int:
        """Magic method for finding size of a component.

        Returns
        -------
        int
            Size of a component.
        """
        return len(self.nodes)

    def __add__(self, other: Component) -> Component:
        """Megic method for merging two components into one big component.

        Parameters
        ----------
        other : Component
            Component to be merged with.

        Returns
        -------
        Component
            A result of merging two smaller components.
        """
        edges = set(chain(self.edges, other.edges, self.connecting_edges(other)))
        nodes = set(chain(self.nodes, other.nodes))

        return Component(
            nodes=nodes,
            edges=edges,
            external_edges=Component._union_external_edges(
                self.external_edges, other.external_edges
            ),
            partition=self.partition,
        )

    @staticmethod
    def _union_external_edges(
        external_edges_1: set[Edge], external_edges_2: set[Edge]
    ) -> set[Edge]:
        all_edges = union_edges(external_edges_1, external_edges_2)
        internal_edges = intersect_edges(external_edges_1, external_edges_2)
        return subtract_edges(all_edges, internal_edges)

    def _count_edges(self, node: tuple[int, int]) -> int:
        return sum([1 for edge in self.edges if node in edge])

    @staticmethod
    def calculate_edge_weight(partition: np.ndarray, edge: Edge) -> float:
        """Find weight of an edge.

        Parameters
        ----------
        partition : np.ndarray
            Image and its partition as a fourth layer.
        edge : Edge
            Edge to calculate weight of.

        Returns
        -------
        float
            Weight of the provided edge.
        """
        return abs(
            np.sum(partition[..., :-1][edge[0]]) / 3
            - np.sum(partition[..., :-1][edge[1]]) / 3
        )

    def boundary_nodes(self) -> set[tuple[int, int]]:
        """Find nodes that are laying on edges of a component.

        Returns
        -------
        set[tuple[int, int]]
            Nodes that are considering to be boundaries of a component.
        """
        _d = {}
        for node in self.nodes:
            _d[node] = self._count_edges(node)
        return {key for key, value in _d.items() if value < 8}

    def connecting_edges(self, other: Component) -> set[Edge]:
        """Find edges connecting to the other component.

        Parameters
        ----------
        other : Component
            A component to search connections with.

        Returns
        -------
        set[Edge]
            Edges that are connecting to the other component.
        """
        component_1_external_edges = self.external_edges
        component_2_external_edges = other.external_edges
        return intersect_edges(component_1_external_edges, component_2_external_edges)

    @cached_property
    def graph(self) -> nx.Graph:
        """Construct a graph from the structure of a component.

        Returns
        -------
        nx.Graph
            Graph built from the component.
        """
        result = nx.Graph()

        for edge in self.edges:
            result.add_edge(
                edge[0],
                edge[1],
                weight=self.calculate_edge_weight(self.partition, edge),
            )

        return result

    @cached_property
    def ind(self) -> float:
        """Find internal difference of the component.

        Returns
        -------
        float
            Internal difference of the component.

        Notes
        -----
        By definition, **internal difference** of a component
        is equal to the maximum weight among all weights of a
        Minimum Spanning Tree of a graph of this component.
        """
        mst: nx.Graph = nx.algorithms.tree.minimum_spanning_tree(
            self.graph, algorithm="prim"
        )

        maximum = 0
        for edge in mst.edges(data=True):

            if edge[2]["weight"] > maximum:
                maximum = edge[2]["weight"]

        return maximum

    @staticmethod
    def exd(
        partition: np.ndarray, component_a: Component, component_b: Component
    ) -> float:
        """Find external difference between components.

        Parameters
        ----------
        partition : np.ndarray
            Image and its partition as a fourth layer.
        component_a : Component
            First component to consider.
        component_b : Component
            Second component to consider.

        Returns
        -------
        float
            External difference between two components.

        Notes
        -----
        By definition, **external difference** (or in some papers just
        **difference**) between Component_a and Component_b is considered
        to be equal the smallest weight of all edges connecting two.
        """
        connecting_edges = component_a.connecting_edges(component_b)

        minimum = np.inf
        for edge in connecting_edges:

            weight = Component.calculate_edge_weight(partition, edge)

            if weight <= minimum:
                minimum = weight

        return minimum
