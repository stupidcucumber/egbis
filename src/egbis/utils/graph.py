import networkx as nx
import numpy as np


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


def generate_graph_from_image(image: np.ndarray, directions: list[tuple]) -> nx.Graph:
    """Generate graph structure from the image.

    Parameters
    ----------
    image : np.ndarray
        Image in the format of (H, W, C) that needs to be represented as a graph.
    directions : list[tuple]
        Directions in which connect pixels to each other in the graph. Possible
        values can be (1, 1), (0, 1), (1, 0), (-1, 0).

    Returns
    -------
    nx.Graph
        Graph representation of an image using NetworkX package.
    """
    graph = nx.Graph()

    for y, row in enumerate(image):
        for x, pixel_value in enumerate(row):
            graph.add_node((y, x), pixel=pixel_value)

            for direction in directions:
                u = (y, x)
                v = (y + direction[0], x + direction[1])

                if is_inside(v, image.shape):
                    graph.add_edge(
                        u, v, weight=abs(intensity(image[u]) - intensity(image[v]))
                    )

    return graph


def calculate_edges_weights(
    partition: np.ndarray, directions: list[tuple[int, int]]
) -> list[tuple[int, int, float]]:
    """Calculate edges weights.

    Parameters
    ----------
    partition : np.ndarray
        Image with partitions as a 4th layer.
    directions : list[tuple[int, int]]
        Directions of the edges to look for.

    Returns
    -------
    list[tuple[int, int, float]]
        Components it connects and the weight itself.
    """
    result = []

    for y_index, row in enumerate(partition):
        for x_index, _ in enumerate(row):

            for direction in directions:

                if is_inside(
                    (y_index + direction[0], x_index + direction[1]), partition.shape
                ):
                    pixel_a = partition[y_index, x_index, :-1]
                    pixel_b = partition[
                        y_index + direction[0], x_index + direction[1], :-1
                    ]
                    result.append(
                        (
                            partition[y_index, x_index, -1],
                            partition[
                                y_index + direction[0], x_index + direction[1], -1
                            ],
                            abs(intensity(pixel_a) - intensity(pixel_b)),
                        )
                    )

    return result


def ndarray_partition_from_image(image: np.ndarray) -> np.ndarray:
    """Generate a partition from image.

    Parameters
    ----------
    image : np.ndarray
        Image to generate partitions from.

    Returns
    -------
    np.ndarray
        Image with partitions as the fourth layer.
    """
    result = np.zeros([*image.shape[:-1], 4])

    result[..., :-1] = image
    result[..., -1] = np.arange(0, np.prod(image.shape[:-1])).reshape(image.shape[:-1])

    return result
