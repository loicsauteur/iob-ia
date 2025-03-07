from bioio import BioImage
import numpy as np
from typing import Any


class CustomImage(BioImage):
    """
    CustomImage is the same as BioImage,
    but adds the get_data function, which the data using np.squeeze
    """

    def __init__(
        self, image,
        reader=None,
        reconstruct_mosaic=True,
        use_plugin_cache=False,
        fs_kwargs={},
        **kwargs: Any
    ):
        super().__init__(
            image=image,
            reader=reader,
            reconstruct_mosaic=reconstruct_mosaic,
            use_plugin_cache=use_plugin_cache,
            fs_kwargs=fs_kwargs,
            **kwargs
        )

    def get_data(self):
        """
        Get the data using np.squeeze.
        Removes axes of dimension = 1, avoiding having T-CZYX dimensions.
        :return:
        """
        return np.squeeze(self.data)
