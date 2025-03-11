import networkx as nx
import numpy as np
from matplotlib.axes import Axes


def draw_segmentation(
    segmentation: list[nx.Graph],
    image: np.ndarray,
    axes: Axes,
    opacity: float    
) -> None:
    segmentation_mask = np.zeros_like(image).astype(np.float32)
    
    for graph in segmentation:
        
        color = np.random.randint(0, 256, 3) / 255
        
        for node in graph.nodes:
            segmentation_mask[node] = color
            
    result_image = (1 - opacity) * image + opacity * segmentation_mask
    
    axes.imshow(result_image)