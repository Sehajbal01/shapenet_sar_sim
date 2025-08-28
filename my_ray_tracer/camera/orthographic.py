# written by JhihYang Wu <jhihyangwu@arizona.edu>

import torch
from ..core import UP

class OrthographicCamera:
    def __init__(self, position, direction, size_w, size_h, pixels_w, pixels_h):
        """
        Constructor for OrthographicCamera class.

        Args:
            position (torch.Tensor): The position of the camera.
            direction (torch.Tensor): Unit vector for the direction the camera is looking at.
            size_w (float): The width of the camera's view.
            size_h (float): The height of the camera's view.
            pixels_w (int): Number of pixels in the w dimension.
            pixels_h (int): Number of pixels in the h dimension.
        """

        self.position = position
        self.direction = direction / torch.norm(direction)  # make sure it's a unit vector
        self.size_w = size_w
        self.size_h = size_h
        self.pixels_w = pixels_w
        self.pixels_h = pixels_h
    
    def generate_rays(self):
        """
        Generate rays for the orthographic camera at current position and direction.

        Returns:
            ray_origins (torch.Tensor): (H, W, 3) tensor of ray origins.
            ray_directions (torch.Tensor): (H, W, 3) tensor of ray directions.
        """
        # generate a grid for sensor
        j_s = torch.linspace(-self.size_w / 2, self.size_w / 2, self.pixels_w)  # negative to positive from left to right of image
        i_s = torch.linspace(self.size_h / 2, -self.size_h / 2, self.pixels_h)  # positive to negative from top to bottom of image
        grid = torch.meshgrid(i_s, j_s, indexing="ij")  # tuple(torch.Size([600, 800]), torch.Size([600, 800]))

        # generate useful vectors
        # same as OpenGL, u vector points to the right from where the sensor is looking, v vector points upward of where the sensor is looking
        u = torch.cross(self.direction, torch.tensor(UP, dtype=torch.float32))
        v = torch.cross(u, self.direction)

        # generate ray origins and directions
        ray_origins = self.position.expand((self.pixels_h, self.pixels_w, 3))  # first let every ray start at camera position
        # apply offset using grid and u v vectors
        ray_origins = ray_origins + grid[1][..., None] * u + grid[0][..., None] * v
        # for orthographic camera, ray direction is all the same and identical to the direction the camera is pointing at
        ray_directions = self.direction.expand((self.pixels_h, self.pixels_w, 3))

        return ray_origins, ray_directions
