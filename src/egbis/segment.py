import logging
import sys

import networkx as nx
import numpy as np
from networkx.algorithms.tree.mst import minimum_spanning_tree

from egbis.utils.graph import calculate_edges_weights, ndarray_partition_from_image

logging.basicConfig(
    level="INFO",
    format="[%(levelname)s] - [%(name)s] - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


def calculate_internal_difference(graph: nx.Graph) -> float:
    """Calculate Ind of a component.

    Parameters
    ----------
    graph : nx.Graph
        Graph representation of a component.

    Returns
    -------
    float
        Ind of a component.
    """
    mst: nx.Graph = minimum_spanning_tree(graph, weight="weight")

    if len(mst.nodes) == 1:
        return 0

    return max(mst.edges(data=True), key=lambda edge: edge[2]["weight"])[2]["weight"]


def find_weight(
    partition: np.ndarray, edge: tuple[tuple[int, int], tuple[int, int]]
) -> float:
    """Calculate weight of an edge.

    Parameters
    ----------
    partition : np.ndarray
        Partition of an image.
    edge : tuple[tuple[int, int], tuple[int, int]]
        Edge for which to calculate weight.

    Returns
    -------
    float
        Weight of an edge.
    """
    pixel_1 = partition[..., :-1][edge[0]]
    pixel_2 = partition[..., :-1][edge[1]]

    return abs(np.sum(pixel_1) / 3 - np.sum(pixel_2) / 3)


def find_nodes(partition: np.ndarray, component_index: int) -> list[tuple[int, int]]:
    """Find nodes in a component.

    Parameters
    ----------
    partition : np.ndarray
        Partition of an image.
    component_index : int
        Index of a component.

    Returns
    -------
    list[tuple[int, int]]
        Nodes found inside the component.
    """
    return [
        tuple(coordinate)
        for coordinate in np.argwhere(partition[..., -1] == component_index)
    ]


def find_internal_edges(
    nodes: list[tuple[int, int]], directions: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    result = []

    for v in nodes:

        for direction in np.asarray(directions):
            u = v + direction

            node_v = tuple(v)
            node_u = tuple(u)
            edge = (node_v, node_u)

            if node_u in nodes and ((node_u, node_v) not in result):

                result.append(edge)

    return result


def find_connecting_edges(
    component_nodes_1: list[tuple[int, int]],
    component_nodes_2: list[tuple[int, int]],
    component_edges_1: list[tuple[tuple[int, int], tuple[int, int]]],
    component_edges_2: list[tuple[tuple[int, int], tuple[int, int]]],
    directions: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:

    def _as_tuples(nodes: np.ndarray) -> list[tuple[int, int]]:
        return [tuple(node) for node in nodes]

    merged_component_edges = find_internal_edges(
        _as_tuples(np.vstack((component_nodes_1, component_nodes_2))),
        directions=directions,
    )

    return set(merged_component_edges).difference(
        set(component_edges_1).union(set(component_edges_2))
    )


def find_minimum_connecting_edge_weight(
    partition: np.ndarray,
    component_nodes_1: list[tuple[int, int]],
    component_nodes_2: list[tuple[int, int]],
    component_edges_1: list[tuple[tuple[int, int], tuple[int, int]]],
    component_edges_2: list[tuple[tuple[int, int], tuple[int, int]]],
    directions: list[tuple],
) -> float:
    connecting_edges = find_connecting_edges(
        component_nodes_1=component_nodes_1,
        component_nodes_2=component_nodes_2,
        component_edges_1=component_edges_1,
        component_edges_2=component_edges_2,
        directions=directions,
    )

    minimum_weight = 1e10
    minimum_edge = None

    for edge in connecting_edges:
        weight = find_weight(partition=partition, edge=edge)

        if weight < minimum_weight:
            minimum_weight = weight
            minimum_edge = edge

    return minimum_edge, minimum_weight


def build_graph_from_nodes(
    partition: np.ndarray,
    nodes: list[tuple[int, int]],
    directions: list[tuple[int, int]],
) -> nx.Graph:
    result = nx.Graph()

    for node in nodes:
        result.add_node(node)

    edges = find_internal_edges(nodes=nodes, directions=directions)

    for edge in edges:
        result.add_edge(
            edge[0], edge[1], weight=find_weight(partition=partition, edge=edge)
        )

    return result


def build_graph_from_nodes_and_edges(
    parttiion: np.ndarray,
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> nx.Graph:
    result = nx.Graph()

    for node in nodes:
        result.add_node(node)

    for edge in edges:
        result.add_edge(
            edge[0], edge[1], weight=find_weight(partition=parttiion, edge=edge)
        )

    return result


def is_mergable(
    partition: np.ndarray,
    component_index_1: int,
    component_index_2: int,
    directions: list[tuple[int, int]],
    k: int,
) -> bool:

    component_nodes_1 = find_nodes(partition, component_index=component_index_1)
    component_nodes_2 = find_nodes(partition, component_index=component_index_2)
    component_edges_1 = find_internal_edges(component_nodes_1, directions=directions)
    component_edges_2 = find_internal_edges(component_nodes_2, directions=directions)

    component_graph_1 = build_graph_from_nodes_and_edges(
        partition, component_nodes_1, component_edges_1
    )
    component_graph_2 = build_graph_from_nodes_and_edges(
        partition, component_nodes_2, component_edges_2
    )

    internal_difference_graph_1 = calculate_internal_difference(component_graph_1)
    internal_difference_graph_2 = calculate_internal_difference(component_graph_2)

    minimal_edge, external_difference = find_minimum_connecting_edge_weight(
        partition=partition,
        component_nodes_1=component_nodes_1,
        component_nodes_2=component_nodes_2,
        component_edges_1=component_edges_1,
        component_edges_2=component_edges_2,
        directions=directions,
    )

    tao_1 = k / len(component_graph_1.nodes)
    tao_2 = k / len(component_graph_2.nodes)

    return not (
        external_difference
        > min(internal_difference_graph_1 + tao_1, internal_difference_graph_2 + tao_2)
    )


def merge(partition: np.ndarray, component_index_1: int, component_index_2: int) -> int:
    component_size_1 = np.sum(partition[..., -1] == component_index_1)
    component_size_2 = np.sum(partition[..., -1] == component_index_2)

    _t = [(component_index_1, component_size_1), (component_index_2, component_size_2)]

    maximum, minimum = sorted(_t, key=lambda item: item[-1])

    partition[..., -1][partition[..., -1] == minimum[0]] = maximum[0]

    return maximum[0]


def update_edge_weights(
    edge_weights: list[tuple[int, int, float]],
    component_index_1: int,
    component_index_2: int,
    merged_component_index: int,
) -> list[tuple[int, int, float]]:

    new_edge_weights = []

    for old_component_index_1, old_component_index_2, weight in edge_weights:

        if (
            old_component_index_1 == component_index_1
            and old_component_index_2 == component_index_2
        ):
            continue

        if (
            old_component_index_1 == component_index_2
            and old_component_index_2 == component_index_1
        ):
            continue

        if old_component_index_1 in [component_index_1, component_index_2]:
            new_edge_weights.append(
                (merged_component_index, old_component_index_2, weight)
            )
        elif old_component_index_2 in [component_index_1, component_index_2]:
            new_edge_weights.append(
                (old_component_index_1, merged_component_index, weight)
            )
        else:
            new_edge_weights.append(
                (old_component_index_1, old_component_index_2, weight)
            )

    new_edge_weights.sort(key=lambda item: item[2])

    return new_edge_weights


def segment(
    image: np.ndarray, iterations: int, directions: list[tuple[int, int]], k: int = 200
) -> np.ndarray:

    partition = ndarray_partition_from_image(image=image)
    edge_weights = calculate_edges_weights(
        partition=partition, directions=[(1, 0), (1, 1), (1, -1), (0, 1)]
    )

    edge_weights.sort(key=lambda item: item[2])

    for _ in range(iterations):

        for component_index_1, component_index_2, _ in edge_weights:

            if not is_mergable(
                partition=partition,
                component_index_1=component_index_1,
                component_index_2=component_index_2,
                directions=directions,
                k=k,
            ):
                continue

            merged_component_index = merge(
                partition=partition,
                component_index_1=component_index_1,
                component_index_2=component_index_2,
            )

            edge_weights = update_edge_weights(
                edge_weights=edge_weights,
                component_index_1=component_index_1,
                component_index_2=component_index_2,
                merged_component_index=merged_component_index,
            )

            break

    return partition
