# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

"""
Input/output utilities for loading and saving data in various formats.
"""

import json
import numpy as np
import cv2 as cv

def load_from_json(filename: str):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.endswith(".json")
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: str, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.endswith(".json")
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)

def read_frame(path):
    """Read a frame from a file."""
    file_format = path.strip().split(".")[-1]
    if file_format in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
        frame = cv.imread(path, cv.IMREAD_UNCHANGED)
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)
    elif path.endswith(".npy"):
        frame = np.load(path)
    else:
        raise NotImplementedError(f"Format {file_format} not supported.")
    return frame

def write_frame(path, frame):
    """Write a frame to a file."""
    file_format = path.strip().split(".")[-1]
    if file_format in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
        cv.imwrite(path, frame)
    elif path.endswith(".npy"):
        np.save(path, frame)
    else:
        raise NotImplementedError(f"Format {file_format} not supported.")
