import itertools

import numpy as np
from tqdm import tqdm

from egbis.types.component import Component, ComponentIndex
from egbis.utils.graph import calculate_edges_weights


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
    tao_1 = k / len(component_1)
    tao_2 = k / len(component_2)

    return not (
        min_connecting_edge_weight
        > min(component_1.ind + tao_1, component_1.ind + tao_2)
    )


def segment(
    image: np.ndarray, directions: list[tuple[int, int]], k: int = 1
) -> dict[ComponentIndex, Component]:
    """Segment image.

    Parameters
    ----------
    image : np.ndarray
        Image to be segmented.
    directions : list[tuple[int, int]]
        Directions of the edges to look for.
    k : int
        Customizable parameter, the higher k the bigger
        components will be. By default, it is 1.

    Returns
    -------
    dict[ComponentIndex, Component]
        Components of the image.
    """
    edges_weights = calculate_edges_weights(image, directions)
    edges_weights.sort(key=lambda item: item[2])

    components: dict[ComponentIndex, Component] = {
        y * image.shape[0]
        + x: Component(nodes={(y, x)}, component_indexes={y * image.shape[0] + x})
        for (y, x) in itertools.product(range(image.shape[0]), range(image.shape[1]))
    }

    for edge_weight in tqdm(edges_weights, total=len(edges_weights)):

        component_index_1, component_index_2, min_connecting_edge_weight = edge_weight

        if not is_mergable(
            component_1=components[component_index_1],
            component_2=components[component_index_2],
            min_connecting_edge_weight=min_connecting_edge_weight,
            k=k,
        ):
            continue

        maximum_index, minimum_index = sorted([component_index_1, component_index_2])

        components[maximum_index].merge(
            components[minimum_index], min_connecting_edge_weight
        )

        for component_index in components[minimum_index].component_indexes:
            components[component_index] = components[maximum_index]

    return components
