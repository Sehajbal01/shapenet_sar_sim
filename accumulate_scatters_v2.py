def accumulate_scatters_from_rays(
    mesh,
    face_normals,
    material_properties,
    rays,
    wavelength=None,
    num_bounce = 1,
    second_bounce_batch_size = 2**100,
    surface_bias = 1e-3,
    ):
    '''
    returns the energy and range for a bunch of rays

    inputs:
        mesh (obj): pytorch3d mesh object of the 3d model
        face_normals (F,3): the normal vector of each face on the mesh
        material_properties (F,5): the r,a,i,d,s of each face of the mesh
        rays (R,3): the locations of the sensor for each pulse for each target scene
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)

    '''
    pass
