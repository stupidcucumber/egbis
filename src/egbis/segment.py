import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Any
from egbis.utils.graph import graph_partition_from_image, ndarray_partition_from_image, calculate_edges_weights
from networkx.algorithms.tree.mst import minimum_spanning_tree
import logging
import sys


logging.basicConfig(
    level="DEBUG",
    format="[%(levelname)s] - [%(name)s] - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


def calculate_internal_difference(graph: nx.Graph) -> float:
    if len(graph.nodes) == 1:
        return 0.0
    
    mst_graph: nx.Graph = minimum_spanning_tree(graph, weight="weight")
    
    maximum_element = max(
        mst_graph.edges(data=True),
        key=lambda x: x[2]["weight"]
    )
    
    return maximum_element[2]["weight"]


def is_mergable(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    edge: tuple[Any, Any, float],
    k: int = 300
) -> bool:
    internal_difference_graph_a = calculate_internal_difference(graph_a)
    internal_difference_graph_b = calculate_internal_difference(graph_b)
    external_difference = edge[2]
    tao_a = k / len(graph_a.nodes)
    tao_b = k / len(graph_b.nodes)
    
    return not (
        external_difference > min(internal_difference_graph_a + tao_a, internal_difference_graph_b + tao_b)
    )


def merge(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    edge: tuple[Any, Any, float]
) -> nx.Graph:
    result = nx.Graph()
    
    result.add_nodes_from(graph_a.nodes(data=True))
    result.add_nodes_from(graph_b.nodes(data=True))
    
    result.add_edges_from(graph_a.edges(data=True))
    result.add_edges_from(graph_b.edges(data=True))
    
    result.add_edge(edge[0], edge[1], weight=edge[2])
    
    return result


def segment(image: np.ndarray, iterations: int = 10, k: int = 600) -> set[nx.Graph]:
    logger.debug(f"Start segmenting image of shape: {image.shape}")
    
    S_partition = graph_partition_from_image(
        image=image, directions=[
            (0, 1), (1, 0), (1, 1)
        ]
    )
    
    logger.debug(f"Partitioned image into {len(S_partition)} components")
    
    for _ in range(iterations):
        S_partition = sorted(S_partition, key=lambda item: item[2][2])

        next_S_partition =  []

        for component_1, component_2, edge in tqdm(S_partition, total=len(S_partition)):

            if is_mergable(component_1, component_2, edge, k=k):

                merged_component = merge(component_1, component_2, edge=edge)
                
                S_partition.remove((component_1, component_2, edge))

                for _component_1, _component_2, _edge in S_partition:
                    
                    if (
                        (_component_1, _component_2) == (component_1, component_2) 
                        or
                        (_component_1, _component_2) == (component_2, component_1)
                    ):
                        
                        if (component_1, component_2, _edge) in S_partition:
                            S_partition.remove(
                                (component_1, component_2, _edge)
                            )
                        
                        if (component_2, component_1, _edge) in S_partition:
                            S_partition.remove(
                                (component_2, component_1, _edge)
                            )

                        merged_component.add_edge(_edge[0], _edge[1], weight=_edge[2])
                        
                        continue
                    
                    if _component_1 == component_1 or _component_1 == component_2:
                        
                        S_partition.remove(
                            (_component_1, _component_2, _edge)
                        )
                        
                        next_S_partition.append(
                            (merged_component, _component_2, _edge)
                        )
                        
                    if _component_2 == component_1 or _component_2 == component_2:
                        
                        S_partition.remove(
                            (_component_1, _component_2, _edge)
                        )
                        
                        next_S_partition.append(
                            (_component_1, merged_component, _edge)
                        )
                
        next_S_partition.extend(S_partition)

        if len(next_S_partition) > 0:
            S_partition = next_S_partition
        else:
            break
    
    result = set()
    
    for component_1, component_2, _ in S_partition:
        result.add(component_1)
        result.add(component_2)
        
    return result


def segment_array_partition(
    image: np.ndarray,
    iterations: int,
    k: int = 200    
) -> np.ndarray:

    partition = ndarray_partition_from_image(image=image)
    edge_weights = calculate_edges_weights(partition=partition, directions=[(1, 0), (1, 1), (0, 1)])
    
    edge_weights.sort(key=lambda item: item[2])

    for _ in range(iterations):
        
        pass
        
    return partition[-1]