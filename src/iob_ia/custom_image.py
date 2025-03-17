from typing import Any, Optional

import dask.array as da
import numpy as np
from bioio import BioImage


class CustomImage(BioImage):
    """
    CustomImage is the same as BioImage.

    It adds the get_data and get_dask_data functions,
    which return the data using np.squeeze.
    """

    def __init__(
        self,
        image,
        reader=None,
        reconstruct_mosaic=True,
        use_plugin_cache=False,
        fs_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Create a BioImage object as intended."""
        if fs_kwargs is None:
            fs_kwargs = {}
        super().__init__(
            image=image,
            reader=reader,
            reconstruct_mosaic=reconstruct_mosaic,
            use_plugin_cache=use_plugin_cache,
            fs_kwargs=fs_kwargs,
            **kwargs,
        )

    def get_data(self) -> np.ndarray:
        """
        Get the data using np.squeeze.

        Remove axes of dimension = 1, avoiding having T-CZYX dimensions.
        :return:
        """
        return np.squeeze(self.data)

    def get_dask_data(self) -> da.Array:
        """
        Get the dask data using np.squeeze.

        Remove axes of dimension = 1, avoiding having T-CZYX dimensions.
        :return:
        """
        return np.squeeze(self.dask_data)
