import itertools
import logging
import sys

import numpy as np
from tqdm import tqdm

from egbis.types.component import Component, Directions, Edge
from egbis.utils.graph import calculate_edges_weights, ndarray_partition_from_image

logging.basicConfig(
    level="INFO",
    format="[%(levelname)s] - [%(name)s] - %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


ComponentIndex = int


def is_mergable(
    component_1: Component,
    component_2: Component,
    min_connecting_edge_weight: float,
    k: int,
) -> bool:
    """Check whether the two components can be merged together.

    Parameters
    ----------
    component_1 : Component
        First component to be considered.
    component_2 : Component
        Second component to be considered.
    min_connecting_edge_weight : float
        Minimum weight among edges connecting components.
    k : int
        Customizable parameter for adjusting degree of attention to
        the size of a component.

    Returns
    -------
    bool
        True if components can be merged, False - otherwise.

    Notes
    -----
    To be considered mergable, the minimum internal difference of
    two components must be bigger than difference between them.
    """
    tao_1 = k / len(component_1.nodes)
    tao_2 = k / len(component_2.nodes)

    return not (
        min_connecting_edge_weight
        > min(component_1.ind + tao_1, component_1.ind + tao_2)
    )


def update_edge_weights_inplace(
    edge_weights: list[tuple[ComponentIndex, ComponentIndex, float]],
    component_index_1: ComponentIndex,
    component_index_2: ComponentIndex,
    merged_component_index: ComponentIndex,
) -> None:
    """Update edges during runtime.

    Parameters
    ----------
    edge_weights : list[tuple[ComponentIndex, ComponentIndex, float]]
        Edges to be updated.
    component_index_1 : ComponentIndex
        Index of the first component to be considered.
    component_index_2 : ComponentIndex
        Index of the second component to be considered.
    merged_component_index : ComponentIndex
        Index of the merged component to be considered.
    """
    for index, edge_weight in enumerate(edge_weights):

        if edge_weight is None:
            continue

        old_component_index_1, old_component_index_2, weight = edge_weight

        if (old_component_index_1, old_component_index_2) in [
            (component_index_1, component_index_2),
            (component_index_2, component_index_1),
        ]:
            edge_weights[index] = None
            continue

        if old_component_index_1 in [component_index_1, component_index_2]:
            edge_weights[index] = (
                merged_component_index,
                old_component_index_2,
                weight,
            )

        elif old_component_index_2 in [component_index_1, component_index_2]:
            edge_weights[index] = (
                old_component_index_1,
                merged_component_index,
                weight,
            )


def initialize_external_edges(
    node: tuple[int, int], directions: list[tuple[int, int]], bounds: tuple[int, int]
) -> set[Edge]:
    """Initialize external edges.

    Parameters
    ----------
    node : tuple[int, int]
        Node that needs an initialization of external edges.
    directions : list[tuple[int, int]]
        Directions in which to search for external edges.
    bounds : tuple[int, int]
        Bounds in which external edges.

    Returns
    -------
    set[Edge]
        External edges for initialization.
    """
    result = set()

    for direction in directions:

        external_node = (node[0] + direction[0], node[1] + direction[1])

        if all(
            [
                external_node[0] >= 0,
                external_node[0] < bounds[0],
                external_node[1] >= 0,
                external_node[1] < bounds[1],
            ]
        ):
            result.add((node, external_node))

    return result


def segment(
    image: np.ndarray, iterations: int, k: int = 1
) -> dict[ComponentIndex, Component]:
    """Segment image into components.

    Parameters
    ----------
    image : np.ndarray
        Image to be segmented.
    iterations : int
        Number of iterations to do.
    k : int
        Customizable parameter for adjusting degree of attention to
        the size of a component. By default, k=1.

    Returns
    -------
    dict[ComponentIndex, Component]
        Dictionary containing indexes of components and
        components themselves.
    """
    partition = ndarray_partition_from_image(image=image)
    edge_weights = calculate_edges_weights(
        partition=partition, directions=[(1, 0), (1, 1), (1, -1), (0, 1)]
    )
    edge_weights.sort(key=lambda item: item[2])

    components: dict[ComponentIndex, Component] = {
        y * image.shape[0]
        + x: Component(
            nodes={(y, x)},
            external_edges=initialize_external_edges(
                (y, x), Directions, image.shape[:-1]
            ),
            partition=partition,
        )
        for (y, x) in itertools.product(range(image.shape[0]), range(image.shape[1]))
    }

    unmergable = set()

    last_number = len(components.keys())
    for iteration in range(iterations):

        logger.info(f"Performing {iteration} iteration...")
        for edge_weight in tqdm(edge_weights, total=len(edge_weights)):

            if edge_weight is None:
                continue

            component_index_1, component_index_2, min_connecting_edge_weight = (
                edge_weight
            )
            component_size_1, component_size_2 = len(
                components[component_index_1]
            ), len(components[component_index_2])

            key = (
                component_index_1,
                component_size_1,
                component_index_2,
                component_size_2,
            )

            if key in unmergable:
                continue

            if not is_mergable(
                component_1=components[component_index_1],
                component_2=components[component_index_2],
                min_connecting_edge_weight=min_connecting_edge_weight,
                k=k,
            ):
                unmergable.add(key)
                continue

            maximum_index, minimum_index = sorted(
                [component_index_1, component_index_2]
            )

            components[maximum_index] = (
                components[maximum_index] + components[minimum_index]
            )

            components.pop(minimum_index)

            update_edge_weights_inplace(
                edge_weights=edge_weights,
                component_index_1=maximum_index,
                component_index_2=minimum_index,
                merged_component_index=maximum_index,
            )

        for index, edge_weight in enumerate(edge_weights):

            if edge_weight is None:
                edge_weights.pop(index)

        if last_number == len(components.keys()):
            break

        last_number = len(components.keys())

    return components
