# written by JhihYang Wu <jhihyangwu@arizona.edu>

import torch

class TriMesh:
    def __init__(self):
        """
        Constructor for the TriMesh class.

        Attributes:
            self.triangles_A (torch.Tensor): (N, 3) tensor of triangle vertices A.
            self.triangles_B (torch.Tensor): (N, 3) tensor of triangle vertices B.
            self.triangles_C (torch.Tensor): (N, 3) tensor of triangle vertices C.
            self.triangles_edge1 (torch.Tensor): (N, 3) tensor of triangle edges 1.
            self.triangles_edge2 (torch.Tensor): (N, 3) tensor of triangle edges 2.
            self.triangle_normal (torch.Tensor): (N, 3) tensor of triangle normals.
        """
        self.triangles_A = None
        self.triangles_B = None
        self.triangles_C = None
        self.triangles_edge1 = None
        self.triangles_edge2 = None
        self.triangles_normal = None

    def load_obj_file(self, filename):
        """
        Loads all the triangles from an obj file.

        Args:
            filename (str): path to the obj file.

        Returns:
            None
        """
        vertex_buffer = []

        self.triangles_A = []
        self.triangles_B = []
        self.triangles_C = []

        with open(filename, "r") as file:
            for line in file:
                line = line.split()
                if len(line) > 0:
                    if line[0] == "v":
                        x, y, z = line[1:]
                        vertex_buffer.append([float(x), float(y), float(z)])
                    elif line[0] == "f":
                        line = line[1:]
                        for v, list in zip(line, [self.triangles_A, self.triangles_B, self.triangles_C]):
                            idx = int(v.split("/")[0])  # only need vertex index, don't need texture index and normal index
                            list.append(vertex_buffer[idx - 1])  # -1 because obj files are 1-indexed but python is 0-indexed

        self.triangles_A = torch.tensor(self.triangles_A, dtype=torch.float32)
        self.triangles_B = torch.tensor(self.triangles_B, dtype=torch.float32)
        self.triangles_C = torch.tensor(self.triangles_C, dtype=torch.float32)

        self.triangles_edge1 = self.triangles_B - self.triangles_A
        self.triangles_edge2 = self.triangles_C - self.triangles_A
        self.triangles_normal = torch.cross(self.triangles_edge1, self.triangles_edge2)
        self.triangles_normal = self.triangles_normal / torch.norm(self.triangles_normal, dim=1, keepdim=True)

    def get_bounds(self):
        """
        Get the axis-aligned bounding box bounds of the mesh.
        
        Returns:
            min_pt (torch.Tensor): (3,) tensor representing the minimum point of the bounding box.
            max_pt (torch.Tensor): (3,) tensor representing the maximum point of the bounding box.
        """
        all_triangle_vertices = torch.cat((self.triangles_A, self.triangles_B, self.triangles_C), dim=0)
        min_pt = torch.min(all_triangle_vertices, dim=0)
        max_pt = torch.max(all_triangle_vertices, dim=0)
        return min_pt.values, max_pt.values

    def to(self, device):
        """
        Move all the data to the specified device.

        Args:
            device (torch.device): the device to move the data to.       
        """
        self.triangles_A = self.triangles_A.to(device)
        self.triangles_B = self.triangles_B.to(device)
        self.triangles_C = self.triangles_C.to(device)
        self.triangles_edge1 = self.triangles_edge1.to(device)
        self.triangles_edge2 = self.triangles_edge2.to(device)
        self.triangles_normal = self.triangles_normal.to(device)

        return self
