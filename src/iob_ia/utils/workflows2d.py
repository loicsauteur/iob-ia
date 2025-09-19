"""Collection of project specific workflows."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from iob_ia.utils.filter_labels import (
    filter_labels_by_property,
    measure_label_props,
)
from iob_ia.utils.io_utils import (
    gen_out_path,
    read_image,
    save_image_channel,
    save_labels,
)
from iob_ia.utils.measure import overlapping_labels
from iob_ia.utils.segment import segment_2d_sub_projections


def calc_intensity_threshold(
    props: dict, channel_name: str = "", verbose: bool = False
) -> float:
    """
    Calculates the intensity threshold based on mean value (divided by 3).

    This is empirical...
    :param props: dict of regionprops_table
    :param channel_name: name of the channel for verbose output
    :param verbose: whether to print out the information
    :return: float of intensity_mean threshold
    """
    intensity_mean = props.get("intensity_mean")
    t = np.sum(intensity_mean) / len(intensity_mean)
    if verbose:
        if channel_name != "":
            print(
                f"For channel '{channel_name}': Average of object mean "
                f"intensities: {t}. Found threshold (divided by 3) = {t / 3}"
            )
        else:
            print(
                f"Average of object mean intensities: {t}. "
                f"Found threshold (divided by 3) = {t / 3}"
            )
    return t / 3


class Iob2DImageCheck:
    """
    Class for checking 3D images using a 2D segmentation approach.

    This class takes care of opening an image and allows setting of segmentation
    parameters.
    It segments the 3D image using:
    - maximum intensity projection of 40 micro meter substack
    - Filtering of objects on size (hard-coded) and mean intensity (empirical, 1/3 of
      the average of the object mean intensities)
    - Checks for double positivity if 2 channels are requested for segmentation.
    """

    def __init__(
        self, path: str, segment_channels: list[int], voxel_size=None
    ):
        """
        Initialize the object.

        Provide the image path and the channels to segment. voxel_size is optional.
        :param path: str path to image file
        :param segment_channels: list of 0-based channel indexes (to be segmented)
        :param voxel_size: tuple for ZYX voxel size
        """
        self.path = path
        self.segment_channels = segment_channels
        self.voxel_size = voxel_size
        # load image
        self.bioimage = read_image(self.path)
        self.channel_names = self.bioimage.channel_names
        # check the voxel size information
        self.__check_metadata__()
        # check image shape
        self.__check_shape__()
        # check channels to segment
        self.__check_channels_to_segment__()

        # Initialize other field variables
        self.original_img = None
        self.imgs = None  # stack of maxIP substacks
        self.modified_voxels_size = None
        self.masks = None
        self.filtered_masks = None
        self.double_pos_masks = None
        self.result_table = None

    def __check_metadata__(self):
        """
        Check if metadata (voxel size is correct).

        Needs to be a tuple of length 3
        :return:
        """
        if self.voxel_size is None:
            self.voxel_size = self.bioimage.physical_pixel_sizes
            if None in self.voxel_size:
                # set the field voxel size to None
                self.voxel_size = None
                warnings.warn(
                    "Voxel size is missing. Image metadata is = "
                    f"{self.bioimage.physical_pixel_sizes}."
                    "Please provide the voxel-size via set_voxel_size((Z, Y, X)).",
                    stacklevel=2,
                )
        elif len(self.voxel_size) != 3:
            raise ValueError(
                "Voxel size must be a tuple of length 3. "
                f"You have {len(self.voxel_size)} elements. "
                f"Use set_voxel_size((Z, Y, X))."
            )

    def set_voxel_size(self, voxel_size: tuple[float]):
        """
        Set the voxel size of the image.

        Basically re-runs the __init__ with the provided new voxel size.
        :param voxel_size:
        :return:
        """
        if len(voxel_size) != 3:
            raise ValueError(
                "Voxel size must be a tuple of length 3. "
                f"You have {len(voxel_size)} elements. Use set_voxel_size((Z, Y, X))."
            )
        self.__init__(self.path, self.segment_channels, voxel_size)

    def __check_shape__(self):
        """
        Check if the image is 3D or 4D.

        Also sets the shape field, after removing dimensions of size 1.
        :return:
        """
        shape = list(self.bioimage.shape)
        while 1 in shape:
            shape.remove(1)
        self.shape = tuple(shape)
        self.ndim = len(shape)
        if self.ndim < 3 or self.ndim > 4:
            self.shape = None
            self.ndim = None
            self.bioimage = None
            raise RuntimeError(
                "Image must have 3 or 4 dimensions. "
                f"You have {self.ndim} dimensions. Please use an 3D or 4D image."
            )

    def __check_channels_to_segment__(self):
        """
        Check if requested channels to segment are valid.

        Among others, checks if the requested channels are 0-based inputs.
        :return:
        """
        # Sanity check
        if self.ndim is None:
            raise RuntimeError("Invalid image. Please use an 3D or 4D image.")
        if self.ndim == 3 and len(self.segment_channels) != 1:
            self.segment_channels = [0]
            raise RuntimeWarning(
                "You 3D image has only one channel. "
                f"But you asked to segment {len(self.segment_channels)} channels. "
                "Ignoring the request and continue segmenting the only channel."
            )
        elif self.ndim == 3 and len(self.segment_channels) == 1:
            if self.segment_channels[0] != 0:
                # no warning here just ignoring input...
                self.segment_channels = [0]
        else:
            for ch in self.segment_channels:
                if ch + 1 > self.shape[0]:
                    seg_ch = self.segment_channels
                    warnings.warn(
                        "You requested segmentations for channels "
                        f"{seg_ch}. But your image has only "
                        f"{self.shape[0]} channels. Set the channels to segment "
                        "using 0-indexing for channel numbers. With e.g. "
                        "set_segment_channels([1, 2]) for the last 2 "
                        "channels of a 3-channel image.",
                        stacklevel=2,
                    )

    def set_segment_channels(self, channels: list[int]):
        """
        Set the channels to segment.

        Basically re-runs the __init__ with the provided new channel list.
        :param channels: list of 0-based channel indices to segment
        :return:
        """
        self.__init__(self.path, channels, self.voxel_size)

    def do_segment(self):
        """
        Segment images on requested channels.

        Segments with cellpose in 2D of substack maxIPs.
        :return:
        """
        print("Loading image to memory...")
        self.original_img = self.bioimage.get_data()
        self.masks = []
        self.imgs = []
        if self.ndim == 3:
            mask, img, new_voxels_size = segment_2d_sub_projections(
                self.bioimage, self.voxel_size
            )
            self.masks.append(mask)
            self.imgs.append(img)
            self.modified_voxels_size = new_voxels_size
        else:
            for ch in self.segment_channels:
                mask, img, new_voxels_size = segment_2d_sub_projections(
                    self.bioimage.get_data()[ch], self.voxel_size
                )
                self.masks.append(mask)
                self.imgs.append(img)
                self.modified_voxels_size = new_voxels_size

    def do_filter(self, filters: Optional[list[dict]] = None):
        """
        Basic filtering of the mask objects.

        Uses hardcoded sizes, but dynamic mean intensity thresholds (average of
        intensity_mean / 3)
        :return:
        """
        if self.masks is None:
            raise RuntimeError("Segmentation must be done before filtering.")
        if filters is not None:
            if len(self.segment_channels) != len(filters):
                raise RuntimeError(
                    "The number of filter dictionaries must be the same "
                    "as the number of channels to segment."
                )
        else:
            # Define the filters
            filters = [
                {
                    "projected_area": [42, 233],  # approx. in pixels:100-550
                    "intensity_mean": [None, None],
                }
            ] * len(self.masks)

        self.filtered_masks = []
        mask_props = []  # currently unused
        filtered_props = []  # currently unused
        for i in range(len(self.masks)):
            print(
                ">>Measuring properties of channel:", self.segment_channels[i]
            )
            props = measure_label_props(
                img_label=self.masks[i],
                img_intensity=self.imgs[i],
                voxel_size=self.voxel_size,
                verbose=False,
            )
            mask_props.append(props)
            print(">>Filtering...")
            # modify the filter for mean intensity
            filters[i]["intensity_mean"] = [
                calc_intensity_threshold(
                    props,
                    channel_name=self.channel_names[self.segment_channels[i]],
                    verbose=True,
                ),
                None,
            ]
            # Filter
            filtered_mask, filtered_prop = filter_labels_by_property(
                self.masks[i], props, filters[i], verbose=True
            )
            self.filtered_masks.append(filtered_mask)
            filtered_props.append(filtered_prop)

    def check_double_pos(self):
        """
        Check for double positivity.

        Currently only supported for comparing 2 channels.
        Creates double_pos_masks, first for channel1 objects that have
        min. 50% overlap with objects in channel2. Then vice-versa.
        :return:
        """
        if len(self.filtered_masks) == 1:
            # In case analysis was run already with 2 channels
            if self.double_pos_masks is not None:
                self.double_pos_masks = None
            return
        elif len(self.filtered_masks) != 2:
            raise NotImplementedError(
                "Checking double-positivity is only supported for comparing 2 channels."
            )
        # get labels of img1 that have min 50% overlap with labels in img2
        self.double_pos_masks = []
        self.double_pos_masks.append(
            overlapping_labels(self.filtered_masks[0], self.filtered_masks[1])
        )
        self.double_pos_masks.append(
            overlapping_labels(self.filtered_masks[1], self.filtered_masks[0])
        )

    def calculate_results(self):
        """
        Count the segmentation output.

        Puts the results in a pandas dataframe.
        :return:
        """
        n_cellpose_objects = []
        n_filtered_objects = []
        n_double_pos_objects = []
        descriptions = []
        for ch in range(len(self.segment_channels)):
            descriptions.append(
                f"Channel {self.segment_channels[ch]}: "
                f"{self.channel_names[self.segment_channels[ch]]}"
            )
            if self.masks is not None:
                n_cellpose_objects.append(len(np.unique(self.masks[ch])) - 1)
            if self.filtered_masks is not None:
                n_filtered_objects.append(
                    len(np.unique(self.filtered_masks[ch])) - 1
                )
            if self.double_pos_masks is not None:
                n_double_pos_objects.append(
                    len(np.unique(self.double_pos_masks[ch])) - 1
                )
            else:
                n_double_pos_objects = ["n/a"] * len(self.segment_channels)

        table = {
            "Channel": descriptions,
            "Cellpose objects": n_cellpose_objects,
            "Filtered objects": n_filtered_objects,
            "Double-positive objects": n_double_pos_objects,
        }
        self.result_table = pd.DataFrame.from_dict(table)
        print("-------------- Results --------------")
        print(self.result_table.to_string())
        if n_double_pos_objects[0] != "n/a":
            print(
                f"{descriptions[0]} 'double-positve objects'= {descriptions[0]} "
                f"objects that have 50% overlap with {descriptions[1]} objects."
            )
            print(
                f"{descriptions[1]} 'double-positve objects'= {descriptions[1]} "
                f"objects that have 50% overlap with {descriptions[0]} objects."
            )
        print("-------------------------------------")

    def analyse(self):
        """
        Run the analysis.

        Combines segmentation, filtering, double positivity check and object counting.
        :return:
        """
        self.do_segment()
        self.do_filter()
        self.check_double_pos()
        self.calculate_results()

    def save_results(self, exclude_cp_masks: bool = False):
        """
        Save mask, channel images and results to file.

        Save the mask images (compressed) to the same folder as the original image,
        as tif files.
        Double positive mask saving is semi-hardcoded.
        :param exclude_cp_masks:
        :return:
        """
        wf = "IOB-2D-workflow"  # workflow identifier for output file names
        if not exclude_cp_masks and self.masks is not None:
            for i in range(len(self.segment_channels)):
                out_path = gen_out_path(
                    self.path,
                    name=f"{wf}_cp-mask_ch{self.segment_channels[i]}-"
                    f"{self.channel_names[self.segment_channels[i]]}",
                )
                save_labels(self.masks[i], out_path)
        if self.filtered_masks is not None:
            for i in range(len(self.segment_channels)):
                out_path = gen_out_path(
                    self.path,
                    name=f"{wf}_filtered-mask_ch{self.segment_channels[i]}-"
                    f"{self.channel_names[self.segment_channels[i]]}",
                )
                save_labels(self.filtered_masks[i], out_path)
        if self.double_pos_masks is not None:
            ch1 = f"ch{self.segment_channels[0]}-{self.channel_names[self.segment_channels[0]]}"
            ch2 = f"ch{self.segment_channels[1]}-{self.channel_names[self.segment_channels[1]]}"
            out_path = gen_out_path(
                self.path,
                name=f"{wf}_{ch1}_objects_pos_for_{ch2}_objects_mask",
            )
            save_labels(self.double_pos_masks[0], out_path)
            out_path = gen_out_path(
                self.path,
                name=f"{wf}_{ch2}_objects_pos_for_{ch1}_objects_mask",
            )
            save_labels(self.double_pos_masks[1], out_path)
        if self.imgs is not None:
            for i in range(len(self.segment_channels)):
                out_path = gen_out_path(
                    self.path,
                    name=f"{wf}_image_ch{self.segment_channels[i]}-"
                    f"{self.channel_names[self.segment_channels[i]]}",
                )
                save_image_channel(self.imgs[i], out_path)
        if self.result_table is not None:
            out_path = gen_out_path(self.path, name=f"{wf}_results-table")
            out_path = out_path.replace(".tif", ".csv")
            pd.DataFrame.to_csv(self.result_table, out_path)
            print(f"Saved results table to {out_path}.")

    def get_cp_mask(self, channel: int) -> (np.ndarray, tuple):
        """
        Get a cellpose mask, along with the voxel size of this mask.

        :param channel: 0-based index of the channel of interest
                        (of the input multichannel image)
        :return: label image (mask)
        :return: voxel size tuple
        """
        index = None
        try:
            index = self.segment_channels.index(channel)
        except ValueError as err:
            raise ValueError(
                f"No segmentation for (0-based) channel {channel}. "
                "You specified the segmentation of following "
                f"channels: {self.segment_channels}"
            ) from err
        return self.masks[index], self.modified_voxels_size

    def get_filtered_mask(self, channel: int) -> (np.ndarray, tuple):
        """
        Get a filtered mask, along with the voxel size of this mask.

        :param channel: 0-based index of the channel of interest
                        (of the input multichannel image)
        :return: label image (mask)
        :return: voxel size tuple
        """
        index = None
        try:
            index = self.segment_channels.index(channel)
        except ValueError as err:
            raise ValueError(
                f"No segmentation for (0-based) channel {channel}. "
                "You specified the segmentation of following "
                f"channels: {self.segment_channels}"
            ) from err
        return self.filtered_masks[index], self.modified_voxels_size

    def get_double_pos_mask(self, channel: int) -> (np.ndarray, tuple):
        """
        Get a double positive mask, along with the voxel size of this mask.

        :param channel: 0-based index of the channel of interest
                        (of the input multichannel image)
        :return: label image (mask)
        :return: voxel size tuple
        """
        index = None
        try:
            index = self.segment_channels.index(channel)
        except ValueError as err:
            raise ValueError(
                f"No segmentation for (0-based) channel {channel}. "
                "You specified the segmentation of following "
                f"channels: {self.segment_channels}"
            ) from err
        return self.double_pos_masks[index], self.modified_voxels_size

    def get_channel(self, channel: int) -> (np.ndarray, tuple):
        """
        Get a channel of the input image, along with the voxel size of this channel.

        The channel is the stack of the substack maxIPs.
        :param channel: 0-based index of the channel of interest
                        (of the input multichannel image)
        :return: label image (mask)
        :return: voxel size tuple
        """
        index = None
        try:
            index = self.segment_channels.index(channel)
        except ValueError as err:
            raise ValueError(
                f"No segmentation for (0-based) channel {channel}. "
                "You specified the segmentation of following "
                f"channels: {self.segment_channels}"
            ) from err
        return self.imgs[index], self.modified_voxels_size
