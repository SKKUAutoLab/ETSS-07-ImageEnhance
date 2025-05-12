#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``albumentations`` for data augmentation on ``numpy.ndarray`` images."""

from typing import Literal

from albumentations import *
from albumentations.augmentations.geometric import functional as F
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.type_definitions import Targets
from pydantic import BaseModel, Field

from mon.constants import TRANSFORMS

# ----- Blur -----
TRANSFORMS.register(name="advanced_blur", module=AdvancedBlur)
TRANSFORMS.register(name="blur",          module=Blur)
TRANSFORMS.register(name="defocus",       module=Defocus)
TRANSFORMS.register(name="gaussian_blur", module=GaussianBlur)
TRANSFORMS.register(name="glass_blur",    module=GlassBlur)
TRANSFORMS.register(name="median_blur",   module=MedianBlur)
TRANSFORMS.register(name="motion_blur",   module=MotionBlur)
TRANSFORMS.register(name="zoom_blur",     module=ZoomBlur)


# ----- Crop -----
TRANSFORMS.register(name="at_least_one_bbox_random_crop", module=AtLeastOneBBoxRandomCrop)
TRANSFORMS.register(name="bbox_safe_random_crop",         module=BBoxSafeRandomCrop)
TRANSFORMS.register(name="center_crop",                   module=CenterCrop)
TRANSFORMS.register(name="crop",       				      module=Crop)
TRANSFORMS.register(name="crop_and_pad",                  module=CropAndPad)
TRANSFORMS.register(name="crop_non_empty_mask_if_exists", module=CropNonEmptyMaskIfExists)
TRANSFORMS.register(name="random_crop",	                  module=RandomCrop)
TRANSFORMS.register(name="random_crop_from_borders",      module=RandomCropFromBorders)
TRANSFORMS.register(name="random_crop_near_bbox",         module=RandomCropNearBBox)
TRANSFORMS.register(name="random_resized_crop",           module=RandomResizedCrop)
TRANSFORMS.register(name="random_sized_bbox_safe_crop",   module=RandomSizedBBoxSafeCrop)
TRANSFORMS.register(name="random_sized_crop",             module=RandomSizedCrop)


@TRANSFORMS.register(name="crop_patch")
class CropPatch(DualTransform):
	"""Crop a patch of the image according to
	`<https://github.com/ZhendongWang6/Uformer/blob/main/dataset/dataset_denoise.py>__`
	"""
	
	def __init__(self, patch_size: int = 128, p: float = 0.5):
		super().__init__(p=p)
		self.patch_size = patch_size
		self.r = 0
		self.c = 0
	
	def apply(self, img: np.ndarray, r: int, c: int, **params) -> np.ndarray:
		return img[r:r + self.patch_size, c:c + self.patch_size, :]
	
	def apply_to_mask(self, img: np.ndarray, r: int, c: int, **params) -> np.ndarray:
		return img[r:r + self.patch_size, c:c + self.patch_size, :]
	
	@property
	def targets_as_params(self) -> list[str]:
		return ["image"]
	
	def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
		image   = params["image"]
		h, w, c = image.shape
		if h - self.patch_size == 0:
			r = 0
			c = 0
		else:
			r = np.random.randint(0, h - self.patch_size)
			c = np.random.randint(0, w - self.patch_size)
		return {"r": r, "c": c}


# ----- Domain Adaptation -----
TRANSFORMS.register(name="fda",                           module=FDA)
TRANSFORMS.register(name="histogram_matching",            module=HistogramMatching)
TRANSFORMS.register(name="pixel_distribution_adaptation", module=PixelDistributionAdaptation)


# ----- Dropout -----
TRANSFORMS.register(name="channel_dropout", module=ChannelDropout)
TRANSFORMS.register(name="coarse_dropout",  module=CoarseDropout)
TRANSFORMS.register(name="grid_dropout",    module=GridDropout)
TRANSFORMS.register(name="mask_dropout",    module=MaskDropout)
TRANSFORMS.register(name="xy_masking",      module=XYMasking)


# ----- Geometric -----
# ----- Resize -----
TRANSFORMS.register(name="longest_max_size",  module=LongestMaxSize)
TRANSFORMS.register(name="random_scale",      module=RandomScale)
TRANSFORMS.register(name="resize",            module=Resize)
TRANSFORMS.register(name="smallest_max_size", module=SmallestMaxSize)


@TRANSFORMS.register(name="resize_multiple_of")
class ResizeMultipleOf(DualTransform):
	"""Resize the input to the given height and width and ensure that they are
	constrained to be multiple of a given number.
	
    Args:
        height: Desired height of the output.
        width: Desired width of the output.
        keep_aspect_ratio: If ``True``, keep the aspect ratio of the input sample.
            Output sample might not have the given width and height, and
            resize behaviour depends on the parameter `resize_method`.
            Default: ``False``.
        multiple_of: Output height and width are constrained to be
            multiple of this parameter. Default: ``1``.
        resize_method: Resize method.
            ``"lower_bound"``: Output will be at least as large as the given
                size.
            ``"upper_bound"``: Output will be at max as large as the given size.
                (Output size might be smaller than given size.)
            ``"minimal"``    : Scale as least as possible. (Output size might
                be smaller than given size.)
            Default: ``"lower_bound"``.
        interpolation: Flag that is used to specify the interpolation algorithm.
            One of: ``'cv2.INTER_NEAREST'``, ``'cv2.INTER_LINEAR'``,
            ``'cv2.INTER_CUBIC'``, ``'cv2.INTER_AREA'``, ``'cv2.INTER_LANCZOS4'``.
            Default: ``'cv2.INTER_LINEAR'``.
        p: Probability of applying the transform. Default: 1.
	
    Targets:
        image, mask, bboxes, keypoints.
	
    Image types:
        uint8, float32
    """
	
	_targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)
	
	class InitSchema(BaseModel):
		height           : int   = Field(ge=1,          description="Desired height of the output.")
		width            : int   = Field(ge=1,          description="Desired width of the output.")
		keep_aspect_ratio: bool  = Field(False,         description="Keep the aspect ratio of the input sample.")
		multiple_of      : int   = Field(1,             description="Output height and width are constrained to be multiple of this parameter.")
		resize_method	 : str   = Field("lower_bound", description="Resize method.")
		interpolation    : int   = cv2.INTER_AREA
		p                : float = 1
	
	def __init__(
		self,
		height           : int,
		width            : int,
		keep_aspect_ratio: bool  = False,
		multiple_of      : int   = 1,
		resize_method    : Literal["lower_bound", "upper_bound", "minimal"] = "lower_bound",
		interpolation    : int   = cv2.INTER_AREA,
		p                : float = 1,
	):
		super().__init__(p=p)
		self.height            = height
		self.width             = width
		self.keep_aspect_ratio = keep_aspect_ratio
		self.multiple_of       = multiple_of
		self.resize_method     = resize_method
		self.interpolation     = interpolation
	
	def constrain_to_multiple_of(self, x, min_val: int = 0, max_val: int = None):
		y = (np.round(x / self.multiple_of) * self.multiple_of).astype(int)
		if max_val and y > max_val:
			y = (np.floor(x / self.multiple_of) * self.multiple_of).astype(int)
		if y < min_val:
			y = (np.ceil(x / self.multiple_of) * self.multiple_of).astype(int)
		return y
	
	def get_size(self, height: int, width: int) -> tuple[int, int]:
		# Determine new height and width
		scale_height = self.height / height
		scale_width  = self.width  / width
		
		if self.keep_aspect_ratio:
			if self.resize_method == "lower_bound":
				# Scale such that output size is lower bound
				if scale_width > scale_height:
					# Fit width
					scale_height = scale_width
				else:
					# Fit height
					scale_width = scale_height
			elif self.resize_method == "upper_bound":
				# Scale such that output size is upper bound
				if scale_width < scale_height:
					# Fit width
					scale_height = scale_width
				else:
					# Fit height
					scale_width = scale_height
			elif self.resize_method == "minimal":
				# Scale as least as possible
				if abs(1 - scale_width) < abs(1 - scale_height):
					# Fit width
					scale_height = scale_width
				else:
					# Fit height
					scale_width = scale_height
			else:
				raise ValueError(f"`resize_method` {self.resize_method} not implemented")
		
		if self.resize_method == "lower_bound":
			new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.height)
			new_width  = self.constrain_to_multiple_of(scale_width  * width,  min_val=self.width)
		elif self.resize_method == "upper_bound":
			new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.height)
			new_width  = self.constrain_to_multiple_of(scale_width  * width,  max_val=self.width)
		elif self.resize_method == "minimal":
			new_height = self.constrain_to_multiple_of(scale_height * height)
			new_width  = self.constrain_to_multiple_of(scale_width  * width)
		else:
			raise ValueError(f"resize_method {self.resize_method} not implemented")
		
		return new_height, new_width
	
	def apply(self, img: np.ndarray, interpolation: int, **params: Any) -> np.ndarray:
		height, width = self.get_size(img.shape[0], img.shape[1])
		return F.resize(img, height=height, width=width, interpolation=interpolation)
	
	def apply_to_bbox(self, bbox: np.ndarray, **params: Any) -> np.ndarray:
		# Bounding box coordinates are scale invariant
		return bbox
	
	def apply_to_keypoint(self, keypoint: np.ndarray, **params: Any) -> np.ndarray:
		height, width = self.get_size(params["rows"], params["cols"])
		scale_x       = self.width  / width
		scale_y       = self.height / height
		return F.keypoint_scale(keypoint, scale_x, scale_y)
	
	def get_transform_init_args_names(self) -> tuple[str, ...]:
		return "height", "width", "interpolation"


# ----- Rotate -----
TRANSFORMS.register(name="random_rotate_90", module=RandomRotate90)
TRANSFORMS.register(name="rotate",           module=Rotate)
TRANSFORMS.register(name="safe_rotate",      module=SafeRotate)

# ----- Transform -----
TRANSFORMS.register(name="affine",              module=Affine)
TRANSFORMS.register(name="d4",                  module=D4)
TRANSFORMS.register(name="elastic_transform",   module=ElasticTransform)
TRANSFORMS.register(name="grid_distortion",     module=GridDistortion)
TRANSFORMS.register(name="grid_elastic_deform", module=GridElasticDeform)
TRANSFORMS.register(name="horizontal_flip",     module=HorizontalFlip)
TRANSFORMS.register(name="optical_distortion",  module=OpticalDistortion)
TRANSFORMS.register(name="pad",                 module=Pad)
TRANSFORMS.register(name="pad_if_needed",       module=PadIfNeeded)
TRANSFORMS.register(name="perspective",         module=Perspective)
TRANSFORMS.register(name="piecewise_affine",    module=PiecewiseAffine)
TRANSFORMS.register(name="random_grid_shuffle", module=RandomGridShuffle)
TRANSFORMS.register(name="shift_scale_rotate",  module=ShiftScaleRotate)
TRANSFORMS.register(name="thin_plate_spline",   module=ThinPlateSpline)
TRANSFORMS.register(name="transpose",           module=Transpose)
TRANSFORMS.register(name="vertical_flip",       module=VerticalFlip)


# ----- Mixing -----
TRANSFORMS.register(name="overlay_elements", module=OverlayElements)


# ----- Spectrogram -----
TRANSFORMS.register(name="frequency_masking", module=FrequencyMasking)
TRANSFORMS.register(name="time_masking",      module=TimeMasking)
TRANSFORMS.register(name="time_reverse",      module=TimeReverse)


# ----- Text -----
TRANSFORMS.register(name="text_image", module=TextImage)


# ----- Transform -----
TRANSFORMS.register(name="additive_noise",             module=AdditiveNoise)
TRANSFORMS.register(name="auto_contrast",              module=AutoContrast)
TRANSFORMS.register(name="channel_shuffle",            module=ChannelShuffle)
TRANSFORMS.register(name="chromatic_aberration",       module=ChromaticAberration)
TRANSFORMS.register(name="clahe",                      module=CLAHE)
TRANSFORMS.register(name="color_jitter",               module=ColorJitter)
TRANSFORMS.register(name="downscale",                  module=Downscale)
TRANSFORMS.register(name="emboss",                     module=Emboss)
TRANSFORMS.register(name="equalize",                   module=Equalize)
TRANSFORMS.register(name="fancy_pca",                  module=FancyPCA)
TRANSFORMS.register(name="from_float",                 module=FromFloat)
TRANSFORMS.register(name="gauss_noise",                module=GaussNoise)
TRANSFORMS.register(name="he_stain",                   module=HEStain)
TRANSFORMS.register(name="hue_saturation_value",       module=HueSaturationValue)
TRANSFORMS.register(name="illumination",               module=Illumination)
TRANSFORMS.register(name="image_compression",          module=ImageCompression)
TRANSFORMS.register(name="invert_img",                 module=InvertImg)
TRANSFORMS.register(name="iso_noise",                  module=ISONoise)
TRANSFORMS.register(name="lambda",                     module=Lambda)
TRANSFORMS.register(name="morphological",              module=Morphological)
TRANSFORMS.register(name="multiplicative_noise",       module=MultiplicativeNoise)
TRANSFORMS.register(name="normalize",                  module=Normalize)
TRANSFORMS.register(name="pixel_dropout",              module=PixelDropout)
TRANSFORMS.register(name="planckian_jitter",           module=PlanckianJitter)
TRANSFORMS.register(name="plasma_brightness_contrast", module=PlasmaBrightnessContrast)
TRANSFORMS.register(name="plasma_shadow",              module=PlasmaShadow)
TRANSFORMS.register(name="posterize",                  module=Posterize)
TRANSFORMS.register(name="random_brightness_contrast", module=RandomBrightnessContrast)
TRANSFORMS.register(name="random_fog",                 module=RandomFog)
TRANSFORMS.register(name="random_gamma",               module=RandomGamma)
TRANSFORMS.register(name="random_gravel",              module=RandomGravel)
TRANSFORMS.register(name="random_rain",                module=RandomRain)
TRANSFORMS.register(name="random_shadow",              module=RandomShadow)
TRANSFORMS.register(name="random_snow",                module=RandomSnow)
TRANSFORMS.register(name="random_sunflare",            module=RandomSunFlare)
TRANSFORMS.register(name="random_tone_curve",          module=RandomToneCurve)
TRANSFORMS.register(name="rgb_shift",                  module=RGBShift)
TRANSFORMS.register(name="ringing_overshoot",          module=RingingOvershoot)
TRANSFORMS.register(name="salt_and_pepper",            module=SaltAndPepper)
TRANSFORMS.register(name="sharpen",                    module=Sharpen)
TRANSFORMS.register(name="shot_noise",                 module=ShotNoise)
TRANSFORMS.register(name="solarize",                   module=Solarize)
TRANSFORMS.register(name="spatter",                    module=Spatter)
TRANSFORMS.register(name="superpixels",                module=Superpixels)
TRANSFORMS.register(name="to_float",                   module=ToFloat)
TRANSFORMS.register(name="to_gray",                    module=ToGray)
TRANSFORMS.register(name="to_rgb",                     module=ToRGB)
TRANSFORMS.register(name="to_sepia",                   module=ToSepia)
TRANSFORMS.register(name="unsharp_mask",               module=UnsharpMask)


@TRANSFORMS.register(name="normalize_image_mean_std")
class NormalizeImageMeanStd(DualTransform):
	"""Normalize image by given `mean` and `std`."""
	
	def __init__(
		self,
		mean: Sequence[float] = [0.485, 0.456, 0.406],
		std : Sequence[float] = [0.229, 0.224, 0.225],
		p   : float = 1.0,
	):
		super().__init__(p=p)
		self.mean = mean
		self.std  = std
	
	def apply(self, img: np.ndarray, **params) -> np.ndarray:
		return (img - self.mean) / self.std
	
	def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
		return (img - self.mean) / self.std
		

# ----- Transform3D -----
TRANSFORMS.register(name="center_crop_3d",    module=CenterCrop3D)
TRANSFORMS.register(name="coarse_dropout_3d", module=CoarseDropout3D)
TRANSFORMS.register(name="cubic_symmetry",    module=CubicSymmetry)
TRANSFORMS.register(name="pad_3d",            module=Pad3D)
TRANSFORMS.register(name="pad_if_needed_3d",  module=PadIfNeeded3D)
TRANSFORMS.register(name="random_crop_3d",    module=RandomCrop3D)
