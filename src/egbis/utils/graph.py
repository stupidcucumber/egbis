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


def intensity(pixel: np.ndarray, max_value: int = 255) -> float:
    """Calculate intensity of the pixel.

    Parameters
    ----------
    pixel : np.ndarray
        Pixel to calculate intensity of.
    max_value : int
        Value that will be used for noramilzation. By default
        it is 255 (maximum RGB value).

    Returns
    -------
    float
        Intensity of the pixel.
    """
    return np.sum(pixel) / 3 / max_value


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
    random = np.random.random((256, 256, 3))

    for y, row in enumerate(random):
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
