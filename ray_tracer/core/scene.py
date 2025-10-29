# written by JhihYang Wu <jhihyangwu@arizona.edu>

import torch
from ray_tracer.geometry.trimesh import TriMesh
from ray_tracer.accelerator.octree import Octree
from ray_tracer.geometry.bbox import BBox
from ..core import UP, EPSILON

class Scene:
    def __init__(self,
                 obj_filename,
                 device,
                 obj_rsa=(0.3,0.3,0.3),
                 octree_max_depth=2,
                 approx_trig_per_bbox=256):
        self.device = device

        # load obj file into trimesh
        self.trimesh = TriMesh()
        self.trimesh.load_obj_file(obj_filename, obj_rsa)
        self.trimesh = self.trimesh.to(device)

        # build octree to speed up ray tracing
        self.octree = Octree(
            max_depth=octree_max_depth,
            approx_trig_per_bbox=approx_trig_per_bbox,
            mesh=self.trimesh,
            device=device
        )

    def add_ground(self, ground_size=1e3, ground_below=True, ground_rsa=(0.3,0.3,0.3)):
        """
        Hack to add ground without adding / changing code anywhere else.
        
        Args:
            ground_size (float): The size of the ground plane.
            ground_below (bool): Whether to place the ground below the object.
            ground_rsa (tuple): reflectivity, specular, ambient for the ground material.
        
        Returns:
            None
        """
        if ground_below == False:
            raise NotImplementedError("Currently only ground_below=True is supported.")
        # add two triangles as ground to all octree leaf elements
        # needs to be done after octree construction because we want octree to be based on scale of the obj file
        ground_size = 1e3
        min_pt, _ = self.trimesh.get_bounds()
        lowest_y = min_pt[1].item()
        # y is lowest point of obj file, x and z are ground_size
        ground_trig_1 = torch.tensor([[-ground_size, lowest_y, -ground_size],
                                    [-ground_size, lowest_y, ground_size],
                                    [ground_size, lowest_y, ground_size]]).to(self.device)
        ground_trig_2 = torch.tensor([[ground_size, lowest_y, ground_size],
                                    [ground_size, lowest_y, -ground_size],
                                    [-ground_size, lowest_y, -ground_size]]).to(self.device)
        # add them to mesh
        self.trimesh.triangles_A = torch.cat([self.trimesh.triangles_A, ground_trig_1[0:1], ground_trig_2[0:1]], dim=0)
        self.trimesh.triangles_B = torch.cat([self.trimesh.triangles_B, ground_trig_1[1:2], ground_trig_2[1:2]], dim=0)
        self.trimesh.triangles_C = torch.cat([self.trimesh.triangles_C, ground_trig_1[2:3], ground_trig_2[2:3]], dim=0)
        self.trimesh.triangles_edge1 = self.trimesh.triangles_B - self.trimesh.triangles_A  # recompute
        self.trimesh.triangles_edge2 = self.trimesh.triangles_C - self.trimesh.triangles_A
        self.trimesh.triangles_normal = torch.linalg.cross(self.trimesh.triangles_edge1, self.trimesh.triangles_edge2, dim=1)
        self.trimesh.triangles_normal = self.trimesh.triangles_normal / torch.norm(self.trimesh.triangles_normal, dim=1, keepdim=True)
        self.trimesh.triangles_rsa = torch.cat([self.trimesh.triangles_rsa,
                                                torch.tensor(ground_rsa, dtype=torch.float32).repeat(2, 1).to(self.trimesh.triangles_rsa.device)], dim=0)  # add rsa for ground triangles
        # add them to octree
        num_trigs = len(self.trimesh.triangles_A)
        last_two_indices = torch.tensor([num_trigs - 2, num_trigs - 1], device=self.device)
        def add_ground_fn(octree_node):
            if octree_node.is_leaf:
                octree_node.triangles = torch.cat([octree_node.triangles, last_two_indices], dim=0)
            else:
                for child in octree_node.children:
                    if child is not None:
                        add_ground_fn(child)
        add_ground_fn(self.octree)
        # add two octree nodes because lots of primary rays miss the first octree right now
        octree_parent = Octree(
            max_depth=0,
            approx_trig_per_bbox=256,
            mesh=self.trimesh,
            device=self.device,
            bbox=BBox(torch.full((3,), -ground_size, dtype=torch.float32, device=self.device),
                      torch.full((3,), ground_size, dtype=torch.float32, device=self.device)),
            triangles=[]
        )
        octree_parent.is_leaf = False
        octree_second_child = Octree(
            max_depth=0,
            approx_trig_per_bbox=256,
            mesh=self.trimesh,
            device=self.device,
            bbox=BBox(torch.full((3,), -ground_size, dtype=torch.float32, device=self.device),
                      torch.full((3,), ground_size, dtype=torch.float32, device=self.device)),
            triangles=last_two_indices
        )
        octree_parent.children[0] = self.octree
        octree_parent.children[1] = octree_second_child
        # start tracing from octree_parent
        self.octree = octree_parent

    def get_depth_and_diffuse(self, camera):
        """
        Get the depth and diffuse images of the scene for a given camera.

        Args:
            camera (OrthographicCamera): The camera to generate rays from.
        """
        ray_origins, ray_directions = camera.generate_rays()
        ray_origins = ray_origins.to(self.device)
        ray_directions = ray_directions.to(self.device)
        H, W, _ = ray_origins.shape

        # Flatten rays for octree processing
        ray_origins_flat = ray_origins.view(-1, 3)
        ray_directions_flat = ray_directions.view(-1, 3)
        
        ray_hit_times, ray_hit_triangle_ids = self.octree.intersect_rays(ray_origins_flat, ray_directions_flat)

        # create a depth image
        ray_hit_times[ray_hit_times < 0] = torch.max(ray_hit_times)  # set miss to max depth
        ray_hit_times -= torch.min(ray_hit_times)
        ray_hit_times /= torch.max(ray_hit_times)
        ray_hit_times = 1 - ray_hit_times  # invert to make further = darker
        depth_image = (ray_hit_times.reshape(H, W).cpu().detach().numpy() * 255).astype("uint8")

        # create a diffuse image
        diffuse_image = torch.zeros((H*W,), dtype=torch.float32, device=self.device)
        mask = torch.argwhere(ray_hit_triangle_ids != -1).squeeze()
        normals = self.trimesh.triangles_normal[ray_hit_triangle_ids[mask]]
        diffuse_image[mask] = torch.sum(-camera.direction.to(normals.device) * normals, dim=1)
        diffuse_image[mask] = torch.abs(diffuse_image[mask])  # we do not differentiate between front and back side of triangle
        diffuse_image[mask] = torch.clamp(diffuse_image[mask], 0, 1)
        diffuse_image = (diffuse_image.reshape(H, W).cpu().detach().numpy() * 255).astype("uint8")

        return depth_image, diffuse_image

    def get_energy_range_values(self, cameras, num_bounces=1, debug=False):
        """
        Trace rays to compute energy range values for each camera for multiple bounces.

        Args:
            cameras (list[OrthographicCamera]): The cameras to generate rays from.
            num_bounces (int): The number of bounces to simulate.
        
        Returns:
            energy_range_values (list[list[torch.Tensor]]): 2D list where each row corresponds to one camera/pulse
                                                            and each column corresponds to one bounce.
                                                            Each entry is a (N, 2) tensor where N is the number of hits,
                                                            and each row contains (range, energy) values.
        """
        # retval
        energy_range_values = [[None] * num_bounces for _ in range(len(cameras))]  # 2D list where each row corresponds to one camera
                                                                                   # and each column corresponds to one bounce

        # generate and merge camera rays
        all_ray_origins = []  # trace all the cameras rays at the same time for efficiency
        all_ray_directions = []
        camera_ray_numbers = []  # remember the number of rays that belong to each camera
        for camera in cameras:
            ray_origins, ray_directions = camera.generate_rays()
            camera_ray_numbers.append(ray_origins.shape[0] * ray_origins.shape[1])  # h * w initially, but could decrease as rays miss
            all_ray_origins.append(ray_origins.view(-1, 3))  # flatten to (N, 3)
            all_ray_directions.append(ray_directions.view(-1, 3))  # flatten to (N, 3)
        all_ray_origins = torch.cat(all_ray_origins, dim=0).to(self.device)  # (sum(camera_ray_numbers), 3)
        all_ray_directions = torch.cat(all_ray_directions, dim=0).to(self.device)  # (sum(camera_ray_numbers), 3)

        # initialize cumulative distance (range) tracker for each ray
        cumulative_distances = torch.zeros(all_ray_origins.shape[0], device=self.device)
        energy_in = torch.ones(all_ray_origins.shape[0], device=self.device)  # initial energy for each ray

        for bounce_index in range(num_bounces):
            ray_hit_times, ray_hit_triangle_ids = self.octree.intersect_rays(all_ray_origins, all_ray_directions)
            
            # update cumulative distances with the distance traveled in this segment
            hit_mask_all = ray_hit_times >= 0  # rays that hit something
            cumulative_distances[hit_mask_all] += ray_hit_times[hit_mask_all]

            # we can easily lose track of which rays belong to which camera so we are not parallelizing this
            assert sum(camera_ray_numbers) == all_ray_origins.shape[0], "Mismatch in total number of rays."
            for camera_index, camera in enumerate(cameras):
                # get the rays that belong to this camera
                start = sum(camera_ray_numbers[:camera_index])
                end = start + camera_ray_numbers[camera_index]
                hit_mask = ray_hit_times[start:end] >= 0
                camera_ray_origins = all_ray_origins[start:end][hit_mask]
                camera_ray_directions = all_ray_directions[start:end][hit_mask]
                camera_ray_hit_times = ray_hit_times[start:end][hit_mask]
                camera_ray_hit_triangle_ids = ray_hit_triangle_ids[start:end][hit_mask]
                camera_cumulative_distances = cumulative_distances[start:end][hit_mask]
                camera_energy_in = energy_in[start:end][hit_mask]

                # energy
                # 1. compute useful values
                hit_triangle_normals = self.trimesh.triangles_normal[camera_ray_hit_triangle_ids]  # normals of the hit triangles
                hit_triangle_rsa = self.trimesh.triangles_rsa[camera_ray_hit_triangle_ids]  # rsa of the hit triangles
                # make sure normals point in opposite direction of incident ray
                # incident_directions = camera_ray_directions
                hit_triangle_normals[torch.sum(hit_triangle_normals * camera_ray_directions, dim=1) > 0] *= -1
                camera_ray_hit_pos = camera_ray_origins + camera_ray_hit_times.unsqueeze(1) * camera_ray_directions  # 3D position of where the ray hit
                # 2. direction to camera sensor plane
                # for orthographic camera, the direction back to sensor plane is simply the negative camera direction
                direction_to_sensor = -camera.direction.to(camera_ray_hit_pos.device)
                
                # first, find intersection back to camera sensor plane
                # camera sensor plane is at camera.position with normal = camera.direction
                # ray equation: hit_pos + t * direction_to_sensor
                # plane equation: all xyz that satisfies (x,y,z) . normal = camera.position . normal
                #                                        dot(point, camera.direction) = dot(camera.position, camera.direction)
                #                                        dot(hit_pos + t * direction_to_sensor, camera.direction) = dot(camera.position, camera.direction)
                #                                        t = (dot(camera.position, camera.direction) - dot(hit_pos, camera.direction)) / dot(direction_to_sensor, camera.direction)
                # solving:                               t = dot(camera.position - hit_pos, camera.direction) / dot(direction_to_sensor, camera.direction)
                
                camera_pos = camera.position.to(camera_ray_hit_pos.device)
                camera_dir = camera.direction.to(camera_ray_hit_pos.device)
                # calculate t parameter for intersection with sensor plane
                numerator = torch.sum((camera_pos - camera_ray_hit_pos) * camera_dir.unsqueeze(0), dim=1)
                denominator = torch.sum(direction_to_sensor.unsqueeze(0) * camera_dir.unsqueeze(0), dim=1)
                t_intersect = numerator / denominator
                
                camera_ray_hit_mask = (t_intersect > 0)  # just check that intersection is in forward direction, there is no such thing as sensor bounds for radar

                # 4. check if anything is blocking this point from being seen by camera sensor by running octree.intersect_rays again
                # create rays from hit positions towards camera sensor
                rays_to_camera_origins = camera_ray_hit_pos[camera_ray_hit_mask]
                rays_to_camera_directions = direction_to_sensor.expand_as(rays_to_camera_origins)
                rays_to_camera_origins = rays_to_camera_origins + EPSILON * rays_to_camera_directions  # offset a bit to avoid self-intersection
                _, blocked_ids = self.octree.intersect_rays(rays_to_camera_origins, rays_to_camera_directions)
                not_blocked_mask = blocked_ids == -1  # keep energy values that don't get blocked by any triangles

                # 5. compute energy based on angle between surface normal and direction to camera
                # (similar to diffuse computation but for the direction back to camera)
                hit_normals_masked = hit_triangle_normals[camera_ray_hit_mask][not_blocked_mask]
                hit_rsa_masked = hit_triangle_rsa[camera_ray_hit_mask][not_blocked_mask]
                energy = torch.clamp(torch.sum(direction_to_sensor * hit_normals_masked, dim=1), min=0)  # (N,)
                # multiply by material scatter and energy in
                energy = energy * hit_rsa_masked[:, 1] * camera_energy_in[camera_ray_hit_mask][not_blocked_mask]
                
                # range
                # for values going into energy_range_values, add time it takes to get back to camera
                range_values = camera_cumulative_distances[camera_ray_hit_mask][not_blocked_mask] + t_intersect[camera_ray_hit_mask][not_blocked_mask]

                if debug:
                    # add back the missed rays as 0 range 0 energy
                    range_values = camera_cumulative_distances + t_intersect
                    range_values[camera_ray_hit_mask][not_blocked_mask] = 0
                    tmp_energy = energy
                    energy = torch.zeros_like(range_values, dtype=tmp_energy.dtype)
                    combined_mask = torch.zeros_like(camera_ray_hit_mask, dtype=torch.bool)
                    combined_mask[camera_ray_hit_mask] = not_blocked_mask
                    energy[combined_mask] = tmp_energy

                # store energy and range values
                if range_values.shape[0] > 0:
                    energy_range_values[camera_index][bounce_index] = torch.stack((range_values, energy), dim=1)  # shape = (hits, 2)



            # compute all_ray_origins all_ray_directions for next bounce
            # calculate reflected rays for next bounce, only for rays that hit something this bounce
            if bounce_index < num_bounces - 1:  # only calculate if there are more bounces to do
                # Get all hit positions and normals for next bounce
                hit_mask_all = ray_hit_times >= 0  # mask for all rays that hit something
                new_ray_origins = all_ray_origins[hit_mask_all] + ray_hit_times[hit_mask_all].unsqueeze(1) * all_ray_directions[hit_mask_all]
                
                # Get normals of hit triangles
                hit_normals = self.trimesh.triangles_normal[ray_hit_triangle_ids[hit_mask_all]]
                hit_rsa = self.trimesh.triangles_rsa[ray_hit_triangle_ids[hit_mask_all]]
                incident_directions = all_ray_directions[hit_mask_all]
                
                # Flip normals if they are pointing away from the incident ray
                # We want the normal to point towards the incident ray (dot product should be negative)
                dot_product = torch.sum(hit_normals * incident_directions, dim=1)
                flip_mask = dot_product > 0  # normals pointing away from incident ray
                hit_normals[flip_mask] = -hit_normals[flip_mask]
                
                # Calculate reflected directions using: R = I - 2 * (I . N) * N
                # where I is incident direction, N is normal, R is reflected direction
                dot_in = torch.sum(incident_directions * hit_normals, dim=1).unsqueeze(1)
                new_ray_directions = incident_directions - 2 * dot_in * hit_normals
                
                # Update ray arrays for next bounce
                all_ray_origins = new_ray_origins + EPSILON * new_ray_directions  # offset a bit to avoid self-intersection
                all_ray_directions = new_ray_directions
                
                # Update cumulative distances to only keep distances for rays that hit something
                cumulative_distances = cumulative_distances[hit_mask_all]

                # Update energy_in to only keep energies for rays that hit something
                energy_in = energy_in[hit_mask_all] * hit_rsa[:, 0]  # multiply by reflectivity for next bounce
                
                # Update camera_ray_numbers to track how many rays each camera has after hits
                new_camera_ray_numbers = []
                for camera_index in range(len(cameras)):
                    start = sum(camera_ray_numbers[:camera_index])
                    end = start + camera_ray_numbers[camera_index]
                    camera_hits = torch.sum(hit_mask_all[start:end]).item()
                    new_camera_ray_numbers.append(camera_hits)
                camera_ray_numbers = new_camera_ray_numbers
                assert sum(camera_ray_numbers) == all_ray_origins.shape[0], "Mismatch in total number of rays."
            
        return energy_range_values
