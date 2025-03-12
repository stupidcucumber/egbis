import cv2
import graphviz as gz
import networkx as nx
import numpy as np
from matplotlib.axes import Axes


def draw_segmentation(
    segmentation: list[nx.Graph], image: np.ndarray, axes: Axes, opacity: float
) -> None:
    segmentation_mask = np.zeros_like(image).astype(np.float32)

    for graph in segmentation:

        color = np.random.randint(0, 256, 3) / 255

        for node in graph.nodes:
            segmentation_mask[node] = color

    result_image = (1 - opacity) * image + opacity * segmentation_mask

    axes.imshow(result_image)


def draw_partition_segmentation(
    partition: np.ndarray, axes: Axes, opacity: float
) -> None:
    segmentation_mask = partition[..., -1]
    image = partition[..., :-1]

    segmentation_image = np.zeros_like(image)
    colors = {}

    for y_index, y in enumerate(segmentation_mask):
        for x_index, x in enumerate(y):

            if x not in colors:
                generator = np.random.RandomState(seed=int(x))
                colors[x] = generator.randint(0, 255, 3).astype(np.float32)

            segmentation_image[y_index, x_index] = colors[x]

    final_image = ((1 - opacity) * image + opacity * segmentation_image) / 255
    axes.imshow(final_image)


def draw_graph(graph: nx.Graph, axes: Axes) -> None:
    vis_graph = gz.Graph()

    for node in graph.nodes:
        vis_graph.node(f"{node}")

    for edge in graph.edges:
        vis_graph.edge(str(edge[0]), str(edge[1]))

    filepath = vis_graph.render("graph.dot", format="svg", cleanup=True, engine="dot")

    graph_image = cv2.imread(filepath) / 255

    axes.imshow(graph_image)
