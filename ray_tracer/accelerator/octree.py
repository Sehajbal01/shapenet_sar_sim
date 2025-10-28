# written by JhihYang Wu <jhihyangwu@arizona.edu>

import torch
from collections import deque
from ..geometry.bbox import BBox
from ..geometry.triangle import triangles_rays_intersection

class Octree:
    def __init__(self, max_depth, approx_trig_per_bbox, mesh, device, bbox=None, triangles=None):
        """
        Constructor for the Octree class.
        
        Args:
            max_depth (int): Maximum recursive depth of the octree.
            approx_trig_per_bbox (int): Number of triangles per bbox to stop subdividing.
            mesh (TriMesh): The mesh containing triangle data.
            device (torch.device): The device to use for tensor operations.
            bbox (BBox): Bounding box for this octree node.
            triangles (list): List of triangle indices that may intersect this bounding box. These indices index into mesh.triangle_ arrays.
        """
        self.device = device

        if bbox is None:
            # first call, create bbox from mesh
            min_pt, max_pt = mesh.get_bounds()
            bbox = BBox(min_pt, max_pt)
            triangles = torch.arange(mesh.triangles_A.shape[0], dtype=torch.long, device=device)

        self.bbox = bbox
        self.is_leaf = False  # whether this node is a leaf node. leaf nodes stop subdividing and contain triangles. non-leaf nodes don't contain triangles.
        self.children = [None] * 8  # octree has 8 children
        self.triangles = []  # triangle indices that this bbox decides to store
        self.mesh = mesh

        if len(triangles) <= approx_trig_per_bbox or max_depth == 0:
            # This is a leaf node
            self.is_leaf = True
            self.triangles = triangles
        else:
            # Subdivide into 8 children
            self.is_leaf = False
            
            # Calculate midpoint
            mid_pt = (bbox.min_pt + bbox.max_pt) / 2.0
            
            # Create 8 child bounding boxes
            child_idx = 0
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        # Calculate min and max points for this child
                        if x == 0:
                            min_x = bbox.min_pt[0]
                            max_x = mid_pt[0]
                        else:
                            min_x = mid_pt[0]
                            max_x = bbox.max_pt[0]
                            
                        if y == 0:
                            min_y = bbox.min_pt[1]
                            max_y = mid_pt[1]
                        else:
                            min_y = mid_pt[1]
                            max_y = bbox.max_pt[1]
                            
                        if z == 0:
                            min_z = bbox.min_pt[2]
                            max_z = mid_pt[2]
                        else:
                            min_z = mid_pt[2]
                            max_z = bbox.max_pt[2]
                        
                        child_min = torch.tensor([min_x, min_y, min_z], dtype=torch.float32, device=self.device)
                        child_max = torch.tensor([max_x, max_y, max_z], dtype=torch.float32, device=self.device)
                        child_bbox = BBox(child_min, child_max)
                        
                        # find triangles that intersect this child bbox
                        v0s = mesh.triangles_A[triangles]  # (len(triangles), 3)
                        v1s = mesh.triangles_B[triangles]  # (len(triangles), 3)
                        v2s = mesh.triangles_C[triangles]  # (len(triangles), 3)
                        vertices = torch.cat([v0s, v1s, v2s], dim=1)  # (len(triangles), 9)
                        vertices = vertices.view(-1, 3)  # (len(triangles)*3, 3)
                        mask = child_bbox.is_in_box(vertices)  # (len(triangles)*3,)
                        mask = mask.view(-1, 3)  # (len(triangles), 3)
                        mask = torch.any(mask, dim=1)  # (len(triangles),) True if any vertex of the triangle is in the bbox

                        child_triangles = triangles[torch.argwhere(mask).squeeze()]
                        # Create child octree only if there will be triangles in this children
                        if child_triangles.shape[0] > 0:
                            self.children[child_idx] = Octree(
                                max_depth - 1, 
                                approx_trig_per_bbox,
                                mesh,
                                self.device,
                                bbox=child_bbox,
                                triangles=child_triangles
                            )
                        child_idx += 1

    def intersect_rays(self, ray_origins, ray_directions):
        """
        Intersect many rays with this octree.

        Args:
            ray_origins (torch.Tensor): (N, 3) tensor of ray origins.
            ray_directions (torch.Tensor): (N, 3) tensor of ray directions.

        Returns:
            t_mins torch.Tensor: (N,) tensor of minimum intersection times for each ray. -1 if ray did not hit anything.
            t_mins_indices torch.Tensor: (N,) tensor of triangle indices / IDs corresponding to the minimum intersection times.
        """
        t_mins = torch.full((ray_origins.shape[0],), float("inf"), dtype=torch.float32, device=self.device)
        t_mins_indices = torch.full((ray_origins.shape[0],), -1, dtype=torch.long, device=self.device)

        # quite hard to parallelize with pytorch because each ray intersects with different bboxs, also different number of them
        # instead, parallelize over octree nodes/bboxes instead of rays

        nodes_to_visit = deque([(self, torch.arange(ray_origins.shape[0], device=self.device))])  # tuples of (node, ray_indices of rays that are interested in this node)

        while len(nodes_to_visit) > 0:
            current_node, ray_indices = nodes_to_visit.popleft()

            # check if rays intersects this node's bounding box
            ray_origins_2 = ray_origins[ray_indices]
            ray_directions_2 = ray_directions[ray_indices]
            t_intersect = current_node.bbox.intersect(ray_origins_2, ray_directions_2)
            ray_hit_indices = torch.argwhere(t_intersect != -1).squeeze()  # indices of rays that hit this bbox
            # TODO: should we skip rays that already have a closer hit than this bbox intersection (because it found triangle hits in other bboxes)
            if len(ray_hit_indices.shape) == 0 or ray_hit_indices.shape[0] == 0:
                continue  # no rays hit this bbox
            if current_node.is_leaf:
                # narrow down rays to evaluate
                ray_origins_3 = ray_origins_2[ray_hit_indices]
                ray_directions_3 = ray_directions_2[ray_hit_indices]
                # we can intersect all interesting rays with all triangles in the current node in parallel
                ray_hit_times, ray_hit_triangle_ids = triangles_rays_intersection(ray_origins_3, ray_directions_3,
                                            self.mesh.triangles_A[current_node.triangles],
                                            self.mesh.triangles_edge1[current_node.triangles],
                                            self.mesh.triangles_edge2[current_node.triangles],
                                            self.mesh.triangles_normal[current_node.triangles],
                                            current_node.triangles)
                ray_hit_times[ray_hit_times == -1] = float("inf")  # set misses to inf
                
                # Update t_mins and t_mins_indices where we found closer intersections
                closer_hits = ray_hit_times < t_mins[ray_indices[ray_hit_indices]]
                indices_to_update = ray_indices[ray_hit_indices][closer_hits]
                t_mins[indices_to_update] = ray_hit_times[closer_hits]
                t_mins_indices[indices_to_update] = ray_hit_triangle_ids[closer_hits]
            else:
                for child in current_node.children:
                    if child is not None:
                        nodes_to_visit.append((child, ray_indices[ray_hit_indices]))

        t_mins[t_mins == float("inf")] = -1.0  # set rays that didn't hit anything to -1
        return t_mins, t_mins_indices
