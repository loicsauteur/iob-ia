"""
Currently not used.

Was for testing where the memory leak,
which ultimately was not caused by cellpose...

"""

from multiprocessing import Queue
from typing import Optional

import numpy as np

from iob_ia.utils.segment import segment_2d_sub_projections


def worker_2d(
    img: np.ndarray,
    voxel_size: tuple,
    substack_size: float,
    model_path: str,
    cellprob_threshold: float,
    flow_threshold: float,
    queue: Queue,
):
    """
    Function for multiprocessing in 2D workflow.

    FIXME: currently, not used.
    """
    mask, substacks, voxel = segment_2d_sub_projections(
        img=img,
        voxel_size=voxel_size,
        substack_size=substack_size,
        model_path=model_path,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )
    queue.put((mask, substacks, voxel))


def seg_2d_sub_proj(
    paths: dict[str, list[str]],
    voxel_size: tuple,
    chs_2_segment: list[int],
    substack_size: float = 40.0,
    model_path: str = "cpsam",
    cellprob_threshold: Optional[dict] = None,
    flow_threshold: Optional[dict] = None,
):
    """
    Function for multiprocessing in 2D workflow.

    FIXME: currently, not used.
    :param paths:
    :param voxel_size:
    :param chs_2_segment:
    :param substack_size:
    :param model_path:
    :param cellprob_threshold:
    :param flow_threshold:
    :return:
    """
    # Set cellprob_threshold & flow_threshold
    if cellprob_threshold is None:
        cellprob_threshold = {i: 0.0 for i in chs_2_segment}
    if flow_threshold is None:
        flow_threshold = {i: 0.4 for i in chs_2_segment}
