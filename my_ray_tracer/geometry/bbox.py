# written by JhihYang Wu <jhihyangwu@arizona.edu>

import torch
from ..core import EPSILON

class BBox:
    def __init__(self, p0=None, p1=None):
        """
        Constructor for the Bounding Box class.

        Args:
            p0 (torch.Tensor): (3,) tensor representing one corner of the box.
            p1 (torch.Tensor): (3,) tensor representing the opposite corner of the box.
        """
        self.min_pt = torch.zeros(3, dtype=torch.float32)
        self.max_pt = torch.zeros(3, dtype=torch.float32)
        if p0 is not None and p1 is not None:
            self.min_pt = torch.min(p0, p1)
            self.max_pt = torch.max(p0, p1)

    def is_in_box(self, points):
        """
        Check if points are inside this bbox. Works for any number of points.

        Args:
            points (torch.Tensor): (N, 3) tensor of points.
        
        Returns:
            torch.Tensor: (N,) tensor of bools indicating if each point is in the box.
        """

        return ((points >= self.min_pt) & (points <= self.max_pt)).all(dim=1)

    def intersect(self, ray_origins, ray_directions):
        """
        Intersect a lot of rays with this bbox

        Args:
            ray_origins (torch.Tensor): (N, 3) tensor of ray origins.
            ray_directions (torch.Tensor): (N, 3) tensor of ray directions.
        
        Returns:
            torch.Tensor: (N,) tensor of intersection times (-1 if no intersection)
        """
        # assumes axis-aligned bounding boxes (AABBs)
        # https://tavianator.com/2011/ray_box.html
        # https://github.com/JiayinCao/SORT/blob/master/src/math/bbox.h
        # https://github.com/JhihYangWu/miniRT/blob/main/src/mymath/bbox.cpp
        N = ray_origins.shape[0]
        device = ray_origins.device

        tmins = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
        tmaxs = torch.full((N,), float("inf"), dtype=torch.float32, device=device)

        for axis in range(3):
            denom = ray_directions[:, axis]
            mask = denom.abs() > EPSILON
            t1 = (self.min_pt[axis] - ray_origins[:, axis]) / denom
            t2 = (self.max_pt[axis] - ray_origins[:, axis]) / denom
            tmins[mask] = torch.max(tmins[mask], torch.min(t1[mask], t2[mask]))
            tmaxs[mask] = torch.min(tmaxs[mask], torch.max(t1[mask], t2[mask]))

        return torch.where(tmins > tmaxs, torch.tensor(-1.0, dtype=torch.float32), tmins)
