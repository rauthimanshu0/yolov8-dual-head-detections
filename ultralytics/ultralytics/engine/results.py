# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.

    This class provides a foundation for tensor-like objects with device management capabilities,
    supporting both PyTorch tensors and NumPy arrays. It includes methods for moving data between
    devices and converting between tensor types.

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).

    Methods:
        cpu: Return a copy of this object with data moved to CPU memory.
        numpy: Return a copy of this object with data converted to a NumPy array.
        cuda: Return a copy of this object with data moved to GPU memory.
        to: Return a copy of this object with the data moved to the specified device/dtype.

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_tensor = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data: Union[torch.Tensor, np.ndarray], orig_shape: Tuple[int, int]) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (Tuple[int, int]): Original shape of the image in (height, width) format.

        Examples:
            >>> import torch
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the underlying data tensor.

        Returns:
            (Tuple[int, ...]): The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """
        Return a copy of this object with data moved to CPU memory.

        Returns:
            (BaseTensor): A new BaseTensor (or subclass) with the data tensor on CPU.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        if isinstance(self.data, np.ndarray):
            # Already on CPU as numpy; return a shallow copy to keep semantics consistent with .numpy()
            return self.__class__(self.data.copy(), self.orig_shape)
        # Detach to avoid tracking and move to CPU
        return self.__class__(torch.as_tensor(self.data).detach().cpu(), self.orig_shape)

    def numpy(self):
        """
        Return a copy of this object with data converted to a NumPy array.

        Returns:
            (BaseTensor): A new BaseTensor (or subclass) whose data is a NumPy array.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> isinstance(numpy_tensor.data, np.ndarray)
            True
        """
        if isinstance(self.data, np.ndarray):
            # Return a copy to avoid in-place mutations affecting the original
            return self.__class__(self.data.copy(), self.orig_shape)
        return self.__class__(torch.as_tensor(self.data).detach().cpu().numpy(), self.orig_shape)

    def cuda(self):
        """
        Move the tensor to GPU memory.

        Returns:
            (BaseTensor): A new BaseTensor instance with the data moved to CUDA if available.
                          If CUDA is not available, this will raise the usual PyTorch error.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).to("cuda"), self.orig_shape)

    def to(self, *args, **kwargs):
        """
        Return a copy of this object with the data moved to the specified device and/or dtype.

        Args:
            *args (Any): Passed through to torch.Tensor.to().
            **kwargs (Any): Passed through to torch.Tensor.to().

        Returns:
            (BaseTensor): A new BaseTensor instance with data moved/converted accordingly.

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """
        Return the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a new BaseTensor instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | List[int] | torch.Tensor): Index or indices to select from the data tensor.

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx].copy() if isinstance(self.data, np.ndarray) else self.data[idx], self.orig_shape)


class Results(SimpleClass, DataExportMixin):
    """
    A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various
    Ultralytics models, including detection, segmentation, classification, and pose estimation.
    It supports visualization, data export, and various coordinate transformations.

    Attributes:
        orig_img (np.ndarray): The original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input RGB image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Pandas Dataframe.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.
        to_xml: Convert detection results to XML format.
        to_html: Convert detection results to HTML format.
        to_sql: Convert detection results to an SQL-compatible format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # Get the first result
        >>> boxes = result.boxes  # Get the boxes for the first result
        >>> masks = result.masks  # Get the masks for the first result
        >>> for result in results:
        >>>     result.plot()  # Plot detection results
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: Dict[int, str],
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        obb: Optional[torch.Tensor] = None,
        speed: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (np.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (Dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result = results[0]  # Get the first result
            >>> boxes = result.boxes  # Get the boxes for the first result
            >>> masks = result.masks  # Get the masks for the first result

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """
        Return a Results object for a specific index of inference results.

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.

        Returns:
            (Results): A new Results object containing the specified subset of inference results.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> single_result = results[0]  # Get the first result
            >>> subset_results = results[1:4]  # Get a slice of results
        """
        return self._apply("__getitem__", idx)

    def __len__(self) -> int:
        """
        Return the number of detections in the Results object.
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)
        return 0  # fix: return 0 when nothing present

    def update(
        self,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        obb: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
    ):
        """
        Update the Results object with new detection data.
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = Probs(probs)  # fix: keep type consistent
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """
        Apply a function to all non-empty attributes and return a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args (Any): Variable length argument list to pass to the function.
            **kwargs (Any): Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """
        Return a copy of the Results object with all its tensors moved to CPU memory.

        This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb)
        transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.

        Returns:
            (Results): A new Results object with all tensor attributes on CPU memory.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> cpu_result = results[0].cpu()  # Move the first result to CPU
            >>> print(cpu_result.boxes.device)  # Output: cpu
        """
        return self._apply("cpu")

    def numpy(self):
        """
        Convert all tensors in the Results object to numpy arrays.

        Returns:
            (Results): A new Results object with all tensors converted to numpy arrays.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        Notes:
            This method creates a new Results object, leaving the original unchanged. It's useful for
            interoperability with numpy-based libraries or when CPU-based operations are required.
        """
        return self._apply("numpy")

    def cuda(self):
        """
        Move all tensors in the Results object to GPU memory.

        Returns:
            (Results): A new Results object with all tensors moved to CUDA device.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # Move first result to GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # Move each result to GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """
        Move all tensors in the Results object to the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # Move first result to GPU
            >>> result_cpu = results[0].to("cpu")  # Move first result to CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """
        Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf: bool = True,
        line_width: Optional[float] = None,
        font_size: Optional[float] = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: Optional[np.ndarray] = None,
        im_gpu: Optional[torch.Tensor] = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: Optional[str] = None,
        color_mode: str = "class",
        txt_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Plot detection results on an input RGB image.

        For detection with 'extras' (after cls), draw two lines:
          line 1: Class 1 name (or parsed from combined name) - its score
          line 2: Class 2 name (or parsed from combined name) - its score

        Preferred extras: [head1_id, head2_id, p_head1, p_head2]
        Fallback extras: [p_head1, p_head2] (ids inferred from combined name if possible)
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),
            example=names,
        )

        # Helper: given h1_id,h2_id, try to parse per-head names from combined mapping 'species - stage'
        def _names_for_pair(h1_id: int, h2_id: int) -> Tuple[Optional[str], Optional[str]]:
            if isinstance(names, dict) and len(names):
                try:
                    first = names.get(0) if hasattr(names, "get") else list(names.values())[0]
                    if isinstance(first, str) and " - " in first:
                        species0 = first.split(" - ", 1)[0]
                        n2 = 0
                        idx = 0
                        while idx in names and str(names[idx]).startswith(species0 + " - "):
                            n2 += 1
                            idx += 1
                        if n2 > 0:
                            ci = h1_id * n2 + h2_id
                            if ci in names:
                                cn = str(names[ci])
                                if " - " in cn:
                                    a, b = cn.split(" - ", 1)
                                    return a, b
                except Exception:
                    pass
            return None, None

        # Plot Segment results (unchanged)
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect/OBB results with extras-aware labels
        if pred_boxes is not None and show_boxes:
            names_h1 = getattr(self, "names_head1", None)
            names_h2 = getattr(self, "names_head2", None)

            for i, d in enumerate(reversed(pred_boxes)):
                c = int(d.cls)
                base_label = None
                if (not is_obb) and hasattr(d, "conf") and labels:
                    base_label = f"{names[c]} {float(d.conf):.2f}" if conf else names[c]
                final_label = base_label

                if (not is_obb) and hasattr(d, "extra") and labels:
                    ex = d.extra
                    ne = int(ex.shape[-1]) if hasattr(ex, "shape") else 0
                    if ne >= 4:
                        exv = ex.reshape(-1)
                        h1_id, h2_id = int(exv[0]), int(exv[1])
                        p1, p2 = float(exv[2]), float(exv[3])
                        # Resolve names via provided maps or infer from combined mapping
                        h1_name = names_h1.get(h1_id) if isinstance(names_h1, dict) else None
                        h2_name = names_h2.get(h2_id) if isinstance(names_h2, dict) else None
                        if h1_name is None or h2_name is None:
                            a, b = _names_for_pair(h1_id, h2_id)
                            h1_name = h1_name or a or f"Class 1 ({h1_id})"
                            h2_name = h2_name or b or f"Class 2 ({h2_id})"
                        final_label = f"{h1_name} {p1:.2f}\n{h2_name} {p2:.2f}"
                    elif ne >= 2:
                        exv = ex.reshape(-1)
                        p1, p2 = float(exv[0]), float(exv[1])
                        # Try to split combined name for current class
                        c_name = names.get(c, "") if isinstance(names, dict) else ""
                        if isinstance(c_name, str) and " - " in c_name:
                            a, b = c_name.split(" - ", 1)
                            final_label = f"{a} {p1:.2f}\n{b} {p2:.2f}"
                        else:
                            final_label = f"Class 1 {p1:.2f}\nClass 2 {p2:.2f}"

                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    final_label,
                    color=colors(
                        c if color_mode == "class" else (int(d.id.item()) if (not is_obb and getattr(d, "is_track", False)) else i),
                        True,
                    ),
                )

        # Plot Classify and Pose (unchanged)
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))

        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")
        return annotator.im if pil else annotator.result()

    def show(self, *args, **kwargs):
        """
        Display the image with annotated inference results.

        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.

        Args:
            *args (Any): Variable length argument list to be passed to the `plot()` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # Display the first result
            >>> for result in results:
            >>>     result.show()  # Display all results
        """
        self.plot(show=True, *args, **kwargs)

    def save(self, filename: Optional[str] = None, *args, **kwargs) -> str:
        """
        Save annotated inference results image to file.

        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

        Args:
            filename (str | Path | None): The filename to save the annotated image. If None, a default filename
                is generated based on the original image path.
            *args (Any): Variable length argument list to be passed to the `plot` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

        Returns:
            (str): The filename where the image was saved.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save("annotated_image.jpg")
            >>> # Or with custom plot arguments
            >>> for result in results:
            >>>     result.save("annotated_image.jpg", conf=False, line_width=2)
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self) -> str:
        """
        Return a log string for each task in the results, detailing detection and classification outcomes.

        This method generates a human-readable string summarizing the detection and classification results. It includes
        the number of detections for each class and the top probabilities for classification tasks.

        Returns:
            (str): A formatted string containing a summary of the results. For detection tasks, it includes the
                number of detections per class. For classification tasks, it includes the top 5 class probabilities.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        Notes:
            - If there are no detections, the method returns "(no detections), " for detection tasks.
            - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
            - The returned string is comma-separated and ends with a comma and a space.
        """
        probs = self.probs
        if len(self) == 0:
            return "" if probs is not None else "(no detections), "
        if probs is not None:
            return f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes := self.boxes:
            counts = boxes.cls.int().bincount()
            return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts) if n > 0)

    def save_txt(self, txt_file: Union[str, Path], save_conf: bool = False) -> str:
        """
        Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class confidence x_center y_center width height`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

        return str(txt_file)

    def save_crop(self, save_dir: Union[str, Path], file_name: Union[str, Path] = Path("im.jpg")):
        """
        Save cropped detection images to specified directory.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.
            file_name (str | Path): Base filename for the saved cropped images.

        Notes:
            - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_crop(save_dir="path/to/crops", file_name="detection")
        """
        if self.probs is not None:
            LOGGER.warning("Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(self, normalize: bool = False, decimals: int = 5) -> List[Dict[str, Any]]:
        """
        Convert inference results to a summarized dictionary with optional normalization for box coordinates.

        This method creates a list of detection dictionaries, each containing information about a single
        detection or classification result. For classification tasks, it returns the top class and its
        confidence. For detection tasks, it includes class information, bounding box coordinates, and
        optionally mask segments and keypoints.

        Args:
            normalize (bool): Whether to normalize bounding box coordinates by image dimensions.
            decimals (int): Number of decimal places to round the output values to.

        Returns:
            (List[Dict[str, Any]]): A list of dictionaries, each containing summarized information for a single detection
                or classification result. The structure of each dictionary varies based on the task type
                (classification or detection) and available information (boxes, masks, keypoints).

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            >>>     summary = result.summary()
            >>>     print(summary)
        """
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results


class Boxes(BaseTensor):
    """
    A class for managing and manipulating detection boxes.

    Supports additional 'extras' after the standard columns, e.g.:
      - No tracking:      [x1,y1,x2,y2, conf, cls, extra...]
      - With tracking:    [x1,y1,x2,y2, id,  conf, cls, extra...]
    """

    def __init__(self, boxes: Union[torch.Tensor, np.ndarray], orig_shape: Tuple[int, int]) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n >= 6, f"expected at least 6 values but got {n}"  # must include xyxy, conf, cls (and optional id)
        super().__init__(boxes, orig_shape)
        self.orig_shape = orig_shape

        # Early-out: if no detections, assume non-tracking layout and avoid reductions on empty tensors
        if self.data.shape[0] == 0:
            self._id_idx = None
            self._conf_idx = 4
            self._cls_idx = 5
            start_extra = 6 if n > 6 else n
            self.is_track = False
            self._extra_slice = slice(start_extra, n) if n > start_extra else slice(0, 0)
            return

        # Determine layout:
        # Heuristic to detect tracking id column: id is usually integer-like (>1), conf in [0,1].
        self._id_idx = None
        if n >= 7:
            col4 = self.data[:, 4]
            col5 = self.data[:, 5]
            try:
                col4_max = float(col4.max())
                col5_max = float(col5.max())
            except Exception:
                # Fallback if tensor types confuse numpy; move to CPU numpy first
                col4_np = col4.detach().cpu().numpy() if isinstance(col4, torch.Tensor) else col4
                col5_np = col5.detach().cpu().numpy() if isinstance(col5, torch.Tensor) else col5
                col4_max = float(np.max(col4_np)) if col4_np.size else 0.0
                col5_max = float(np.max(col5_np)) if col5_np.size else 0.0
            maybe_id_first = (col4_max > 1.5) and (col5_max <= 1.0)
            if maybe_id_first:
                # [xyxy, id, conf, cls, extras...]
                self._id_idx = 4
                self._conf_idx = 5
                self._cls_idx = 6
                start_extra = 7
            else:
                # [xyxy, conf, cls, extras...]
                self._conf_idx = 4
                self._cls_idx = 5
                start_extra = 6
        else:
            # Exactly 6 columns
            self._conf_idx = 4
            self._cls_idx = 5
            start_extra = 6

        self.is_track = self._id_idx is not None
        self._extra_slice = slice(start_extra, n) if n > start_extra else slice(0, 0)

    @property
    def xyxy(self) -> Union[torch.Tensor, np.ndarray]:
        return self.data[:, :4]

    @property
    def conf(self) -> Union[torch.Tensor, np.ndarray]:
        return self.data[:, self._conf_idx]

    @property
    def cls(self) -> Union[torch.Tensor, np.ndarray]:
        return self.data[:, self._cls_idx]

    @property
    def id(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        return self.data[:, self._id_idx] if self._id_idx is not None else None

    @property
    def extra(self) -> Union[torch.Tensor, np.ndarray]:
        # Shape (N, ne). If no extras, returns an empty (N, 0) view.
        return self.data[:, self._extra_slice]

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> Union[torch.Tensor, np.ndarray]:
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> Union[torch.Tensor, np.ndarray]:
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> Union[torch.Tensor, np.ndarray]:
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    This class extends BaseTensor and provides functionality for handling segmentation masks,
    including methods for converting between pixel and normalized coordinates.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor or array containing mask data.
        orig_shape (tuple): Original image shape in (height, width) format.
        xy (List[np.ndarray]): A list of segments in pixel coordinates.
        xyn (List[np.ndarray]): A list of normalized segments.

    Methods:
        cpu: Return a copy of the Masks object with the mask tensor on CPU memory.
        numpy: Return a copy of the Masks object with the mask tensor as a numpy array.
        cuda: Return a copy of the Masks object with the mask tensor on GPU memory.
        to: Return a copy of the Masks object with the mask tensor on specified device and dtype.

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks: Union[torch.Tensor, np.ndarray], orig_shape: Tuple[int, int]) -> None:
        """
        Initialize the Masks class with detection mask data and the original image shape.

        Args:
            masks (torch.Tensor | np.ndarray): Detection masks with shape (num_masks, height, width).
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Masks
            >>> masks = torch.rand(10, 160, 160)  # 10 masks of 160x160 resolution
            >>> orig_shape = (720, 1280)  # Original image shape
            >>> mask_obj = Masks(masks, orig_shape)
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> List[np.ndarray]:
        """
        Return normalized xy-coordinates of the segmentation masks.

        This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates
        are normalized relative to the original image shape.

        Returns:
            (List[np.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates
                of a single segmentation mask. Each array has shape (N, 2), where N is the number of points in the
                mask contour.

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # Normalized coordinates of the first mask
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> List[np.ndarray]:
        """
        Return the [x, y] pixel coordinates for each segment in the mask tensor.

        This property calculates and returns a list of pixel coordinates for each segmentation mask in the
        Masks object. The coordinates are scaled to match the original image dimensions.

        Returns:
            (List[np.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel
                coordinates for a single segmentation mask. Each array has shape (N, 2), where N is the
                number of points in the segment.

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # Number of masks
            >>> print(xy_coords[0].shape)  # Shape of first mask's coordinates
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation,
    normalization, and confidence values. It supports keypoint detection results with optional visibility
    information.

    Attributes:
        data (torch.Tensor): The raw tensor containing keypoint data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        has_visible (bool): Indicates whether visibility information is available for keypoints.
        xy (torch.Tensor): Keypoint coordinates in [x, y] format.
        xyn (torch.Tensor): Normalized keypoint coordinates in [x, y] format, relative to orig_shape.
        conf (torch.Tensor): Confidence values for each keypoint, if available.

    Methods:
        cpu: Return a copy of the keypoints tensor on CPU memory.
        numpy: Return a copy of the keypoints tensor as a numpy array.
        cuda: Return a copy of the keypoints tensor on GPU memory.
        to: Return a copy of the keypoints tensor with specified device and dtype.

    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 detection, 17 keypoints, (x, y, conf)
        >>> orig_shape = (480, 640)  # Original image shape (height, width)
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # Access xy coordinates
        >>> print(keypoints.conf)  # Access confidence values
        >>> keypoints_cpu = keypoints.cpu()  # Move keypoints to CPU
    """

    def __init__(self, keypoints: Union[torch.Tensor, np.ndarray], orig_shape: Tuple[int, int]) -> None:
        """
        Initialize the Keypoints object with detection keypoints and original image dimensions.

        This method processes the input keypoints tensor, handling both 2D and 3D formats. For 3D tensors
        (x, y, confidence), it masks out low-confidence keypoints by setting their coordinates to zero.

        Args:
            keypoints (torch.Tensor): A tensor containing keypoint data. Shape can be either:
                - (num_objects, num_keypoints, 2) for x, y coordinates only
                - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores
            orig_shape (Tuple[int, int]): The original image dimensions (height, width).

        Examples:
            >>> kpts = torch.rand(1, 17, 3)  # 1 object, 17 keypoints (COCO format), x,y,conf
            >>> orig_shape = (720, 1280)  # Original image height, width
            >>> keypoints = Keypoints(kpts, orig_shape)
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return x, y coordinates of keypoints.

        Returns:
            (torch.Tensor): A tensor containing the x, y coordinates of keypoints with shape (N, K, 2), where N is
                the number of detections and K is the number of keypoints per detection.

        Examples:
            >>> results = model("image.jpg")
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # x, y coordinates of keypoints for first detection

        Notes:
            - The returned coordinates are in pixel units relative to the original image dimensions.
            - If keypoints were initialized with confidence values, only keypoints with confidence >= 0.5 are returned.
            - This property uses LRU caching to improve performance on repeated access.
        """
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return normalized coordinates (x, y) of keypoints relative to the original image size.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or array of shape (N, K, 2) containing normalized keypoint
                coordinates, where N is the number of instances, K is the number of keypoints, and the last
                dimension contains [x, y] values in the range [0, 1].

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Return confidence values for each keypoint.

        Returns:
            (torch.Tensor | None): A tensor containing confidence scores for each keypoint if available,
                otherwise None. Shape is (num_detections, num_keypoints) for batched data or (num_keypoints,)
                for single detection.

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification probabilities.

    This class extends BaseTensor and provides methods for accessing and manipulating
    classification probabilities, including top-1 and top-5 predictions.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor or array containing classification probabilities.
        orig_shape (tuple | None): The original image shape as (height, width). Not used in this class.
        top1 (int): Index of the class with the highest probability.
        top5 (List[int]): Indices of the top 5 classes by probability.
        top1conf (torch.Tensor | np.ndarray): Confidence score of the top 1 class.
        top5conf (torch.Tensor | np.ndarray): Confidence scores of the top 5 classes.

    Methods:
        cpu: Return a copy of the probabilities tensor on CPU memory.
        numpy: Return a copy of the probabilities tensor as a numpy array.
        cuda: Return a copy of the probabilities tensor on GPU memory.
        to: Return a copy of the probabilities tensor with specified device and dtype.

    Examples:
        >>> probs = torch.tensor([0.1, 0.3, 0.6])
        >>> p = Probs(probs)
        >>> print(p.top1)
        2
        >>> print(p.top5)
        [2, 1, 0]
        >>> print(p.top1conf)
        tensor(0.6000)
        >>> print(p.top5conf)
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs: Union[torch.Tensor, np.ndarray], orig_shape: Optional[Tuple[int, int]] = None) -> None:
        """
        Initialize the Probs class with classification probabilities.

        This class stores and manages classification probabilities, providing easy access to top predictions and their
        confidences.

        Args:
            probs (torch.Tensor | np.ndarray): A 1D tensor or array of classification probabilities.
            orig_shape (tuple | None): The original image shape as (height, width). Not used in this class but kept
                for consistency with other result classes.

        Attributes:
            data (torch.Tensor | np.ndarray): The raw tensor or array containing classification probabilities.
            top1 (int): Index of the top 1 class.
            top5 (List[int]): Indices of the top 5 classes.
            top1conf (torch.Tensor | np.ndarray): Confidence of the top 1 class.
            top5conf (torch.Tensor | np.ndarray): Confidences of the top 5 classes.

        Examples:
            >>> import torch
            >>> probs = torch.tensor([0.1, 0.3, 0.2, 0.4])
            >>> p = Probs(probs)
            >>> print(p.top1)
            3
            >>> print(p.top1conf)
            tensor(0.4000)
            >>> print(p.top5)
            [3, 1, 2, 0]
        """
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self) -> int:
        """
        Return the index of the class with the highest probability.

        Returns:
            (int): Index of the class with the highest probability.

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> probs.top1
            2
        """
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self) -> List[int]:
        """
        Return the indices of the top 5 class probabilities.

        Returns:
            (List[int]): A list containing the indices of the top 5 class probabilities, sorted in descending order.

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
            >>> print(probs.top5)
            [4, 3, 2, 1, 0]
        """
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return the confidence score of the highest probability class.

        This property retrieves the confidence score (probability) of the class with the highest predicted probability
        from the classification results.

        Returns:
            (torch.Tensor | np.ndarray): A tensor containing the confidence score of the top 1 class.

        Examples:
            >>> results = model("image.jpg")  # classify an image
            >>> probs = results[0].probs  # get classification probabilities
            >>> top1_confidence = probs.top1conf  # get confidence of top 1 class
            >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
        """
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return confidence scores for the top 5 classification predictions.

        This property retrieves the confidence scores corresponding to the top 5 class probabilities
        predicted by the model. It provides a quick way to access the most likely class predictions
        along with their associated confidence levels.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or array containing the confidence scores for the
                top 5 predicted classes, sorted in descending order of probability.

        Examples:
            >>> results = model("image.jpg")
            >>> probs = results[0].probs
            >>> top5_conf = probs.top5conf
            >>> print(top5_conf)  # Prints confidence scores for top 5 classes
        """
        return self.data[self.top5]


class OBB(BaseTensor):
    """
    A class for storing and manipulating Oriented Bounding Boxes (OBB).

    This class provides functionality to handle oriented bounding boxes, including conversion between
    different formats, normalization, and access to various properties of the boxes. It supports
    both tracking and non-tracking scenarios.

    Attributes:
        data (torch.Tensor): The raw OBB tensor containing box coordinates and associated data.
        orig_shape (tuple): Original image size as (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xywhr (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | np.ndarray): Tracking IDs for each box, if available.
        xyxyxyxy (torch.Tensor | np.ndarray): Boxes in 8-point [x1, y1, x2, y2, x3, y3, x4, y4] format.
        xyxyxyxyn (torch.Tensor | np.ndarray): Normalized 8-point coordinates relative to orig_shape.
        xyxy (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in [x1, y1, x2, y2] format.

    Methods:
        cpu: Return a copy of the OBB object with all tensors on CPU memory.
        numpy: Return a copy of the OBB object with all tensors as numpy arrays.
        cuda: Return a copy of the OBB object with all tensors on GPU memory.
        to: Return a copy of the OBB object with tensors on specified device and dtype.

    Examples:
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """

    def __init__(self, boxes: Union[torch.Tensor, np.ndarray], orig_shape: Tuple[int, int]) -> None:
        """
        Initialize an OBB (Oriented Bounding Box) instance with oriented bounding box data and original image shape.

        This class stores and manipulates Oriented Bounding Boxes (OBB) for object detection tasks. It provides
        various properties and methods to access and transform the OBB data.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array containing the detection boxes,
                with shape (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values.
                If present, the third last column contains track IDs, and the fifth column contains rotation.
            orig_shape (Tuple[int, int]): Original image size, in the format (height, width).

        Attributes:
            data (torch.Tensor | np.ndarray): The raw OBB tensor.
            orig_shape (Tuple[int, int]): The original image shape.
            is_track (bool): Whether the boxes include tracking IDs.

        Raises:
            AssertionError: If the number of values per box is not 7 or 8.

        Examples:
            >>> import torch
            >>> boxes = torch.rand(3, 7)  # 3 boxes with 7 values each
            >>> orig_shape = (640, 480)
            >>> obb = OBB(boxes, orig_shape)
            >>> print(obb.xywhr)  # Access the boxes in xywhr format
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return boxes in [x_center, y_center, width, height, rotation] format.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array containing the oriented bounding boxes with format
                [x_center, y_center, width, height, rotation]. The shape is (N, 5) where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> obb = results[0].obb
            >>> xywhr = obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3, 5])
        """
        return self.data[:, :5]

    @property
    def conf(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return the confidence scores for Oriented Bounding Boxes (OBBs).

        This property retrieves the confidence values associated with each OBB detection. The confidence score
        represents the model's certainty in the detection.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array of shape (N,) containing confidence scores
                for N detections, where each score is in the range [0, 1].

        Examples:
            >>> results = model("image.jpg")
            >>> obb_result = results[0].obb
            >>> confidence_scores = obb_result.conf
            >>> print(confidence_scores)
        """
        return self.data[:, -2]

    @property
    def cls(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return the class values of the oriented bounding boxes.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array containing the class values for each oriented
                bounding box. The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> result = results[0]
            >>> obb = result.obb
            >>> class_values = obb.cls
            >>> print(class_values)
        """
        return self.data[:, -1]

    @property
    def id(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Return the tracking IDs of the oriented bounding boxes (if available).

        Returns:
            (torch.Tensor | np.ndarray | None): A tensor or numpy array containing the tracking IDs for each
                oriented bounding box. Returns None if tracking IDs are not available.

        Examples:
            >>> results = model("image.jpg", tracker=True)  # Run inference with tracking
            >>> for result in results:
            ...     if result.obb is not None:
            ...         track_ids = result.obb.id
            ...         if track_ids is not None:
            ...             print(f"Tracking IDs: {track_ids}")
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes.

        Returns:
            (torch.Tensor | np.ndarray): Rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where N is
                the number of boxes. Each box is represented by 4 points (x, y), starting from the top-left corner and
                moving clockwise.

        Examples:
            >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
            >>> xyxyxyxy = obb.xyxyxyxy
            >>> print(xyxyxyxy.shape)
            torch.Size([1, 4, 2])
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert rotated bounding boxes to normalized xyxyxyxy format.

        Returns:
            (torch.Tensor | np.ndarray): Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2),
                where N is the number of boxes. Each box is represented by 4 points (x, y), normalized relative to
                the original image dimensions.

        Examples:
            >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 random OBBs
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes.shape)
            torch.Size([10, 4, 2])
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format.

        This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in
        xyxy format (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such
        as IoU calculation with non-rotated boxes.

        Returns:
            (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in xyxy format with shape (N, 4), where N
                is the number of boxes. Each row contains [x1, y1, x2, y2] coordinates.

        Examples:
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n-obb.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     obb = result.obb
            ...     if obb is not None:
            ...         xyxy_boxes = obb.xyxy
            ...         print(xyxy_boxes.shape)  # (N, 4)

        Notes:
            - This method approximates the OBB by its minimal enclosing rectangle.
            - The returned format is compatible with standard object detection metrics and visualization tools.
            - The property uses caching to improve performance for repeated access.
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
