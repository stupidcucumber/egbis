import numpy as np
from matplotlib.axes import Axes

from egbis.types.component import Component


def draw_components_partition(
    image: np.ndarray, components: dict[int, Component], axes: Axes, opacity: float
) -> None:
    """Draw segmentation of an image.

    Parameters
    ----------
    image : np.ndarray
        Image that was segmented.
    components : dict[int, Component]
        Segmentation parts of an image.
    axes : Axes
        Axes on which final image will be shown.
    opacity : float
        Opacity of a segmentation mask.
    """
    segmentation_image = np.zeros_like(image)

    for component_index, component in components.items():

        generator = np.random.RandomState(seed=int(component_index))
        color = generator.randint(0, 255, 3).astype(np.float32)
        for node in component.nodes:
            segmentation_image[node] = color

    final_image = ((1 - opacity) * image + opacity * segmentation_image) / 255
    axes.imshow(final_image)
