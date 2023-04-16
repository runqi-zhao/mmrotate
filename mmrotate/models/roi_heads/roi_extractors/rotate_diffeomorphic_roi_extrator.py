import mmcv
import numpy as np
import torch

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor
from rotate_single_level_roi_extractor import RotatedSingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class RotatedDiffeomorphicRoIExtractor(RotatedSingleRoIExtractor):
    """A customized RoI extractor that applies diffeomorphic transformation
    to the RoI features using the provided diffeomorphic matching algorithm.

    Args:
        diffeomorphic_algo (callable): The diffeomorphic matching algorithm to
            apply for RoI feature transformation.
        **kwargs: Keyword arguments for `SingleRoIExtractor`.
    """

    def __init__(self, diffeomorphic_algo, **kwargs):
        super().__init__(**kwargs)
        self.diffeomorphic_algo = diffeomorphic_algo

    def extract_feats(self, img, rois, with_img=True):
        # Extract features from RoI features using SingleRoIExtractor
        feats = super().extract_feats(img, rois, with_img)

        # Calculate center of RoIs
        centers = (rois[:, 1:3] + rois[:, 3:5]) / 2

        # Apply diffeomorphic transformation to RoI features
        transformed_feats = []
        for i in range(len(feats)):
            feat = feats[i].detach().cpu().numpy()
            im_size = img.size[::-1]
            transformed_feat = self.diffeomorphic_algo(img, feat, centers[i], im_size)
            transformed_feat = torch.from_numpy(transformed_feat).to(device=feat.device)
            transformed_feats.append(transformed_feat)

        # Concatenate transformed RoI features and return
        transformed_feats = torch.cat(transformed_feats, dim=0)
        if with_img:
            return transformed_feats, img
        else:
            return transformed_feats


def diffeomorphic_matching(img, feat, center, im_size):
    # Convert RoI coordinates to image coordinates
    roi_size = feat.shape[-2:]
    x = np.array([
        [-roi_size[1] / 2, -roi_size[1] / 2, roi_size[1] / 2, roi_size[1] / 2],
        [-roi_size[0] / 2, roi_size[0] / 2, roi_size[0] / 2, -roi_size[0] / 2]
    ])
    x += np.array(center).reshape(-1, 2)
    x += 0.5
    x[:, 0] = np.clip(x[:, 0], 0, im_size[0] - 1)
    x[:, 1] = np.clip(x[:, 1], 0, im_size[1] - 1)

    # Convert feature to image mask
    mask = BitmapMasks(feat[None, ...], feat.shape[-2:], True)
    polys = PolygonMasks(mask).to_polygons()[0]
    poly_x = np.clip(polys[:, :, 0], 0, im_size[0] - 1)
    poly_y = np.clip(polys[:, :, 1], 0, im_size[1] - 1)

    # Apply diffeomorphic matching algorithm
    transformed_poly = diffeomorphic_algo(poly_x, poly_y, x[:, 0], x[:, 1])
    transformed_mask = PolygonMasks(transformed_poly).to_bitmap(im_size, True)

