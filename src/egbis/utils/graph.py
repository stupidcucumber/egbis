from itertools import product

import numpy as np

from egbis.types.component import ComponentIndex


def is_inside(node: tuple[int, int], shape: tuple[int, int]) -> bool:
    """Check whether node with a this coordinate is inside the image bounds.

    Parameters
    ----------
    node : tuple[int, int]
        Coordinates of the node in the format (y, x).
    shape : tuple[int, int]
        Shape of the image in the format (H, W).

    Returns
    -------
    bool
        True if node is inside image bounds.
    """
    return all(
        [node[0] >= 0 and node[0] < shape[0], node[1] >= 0 and node[1] < shape[1]]
    )


def intensity(pixel: np.ndarray) -> float:
    """Calculate intensity of the pixel.

    Parameters
    ----------
    pixel : np.ndarray
        Pixel to calculate intensity of.

    Returns
    -------
    float
        Intensity of the pixel.
    """
    return np.sum(pixel) / 3


def calculate_edges_weights(
    image: np.ndarray, directions: list[tuple[int, int]]
) -> list[tuple[ComponentIndex, ComponentIndex, float]]:
    """Calculate edges weights.

    Parameters
    ----------
    image : np.ndarray
        Image to segment.
    directions : list[tuple[int, int]]
        Directions of the edges to look for.

    Returns
    -------
    list[tuple[ComponentIndex, ComponentIndex, float]]
        Components it connects and the edge weight itself.
    """
    result = []

    height, width = image.shape[:-1]

    for y, x in product(range(height), range(width)):

        for delta_y, delta_x in directions:

            other_y = y + delta_y
            other_x = x + delta_x

            if not is_inside((other_y, other_x), (height, width)):
                continue

            result.append(
                (
                    y * height + x,
                    other_y * height + other_x,
                    abs(
                        intensity(pixel=image[y, x])
                        - intensity(pixel=image[other_y, other_x])
                    ),
                )
            )

    return result
