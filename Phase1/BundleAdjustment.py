
import numpy as np
import cv2
from scipy.optimize import least_squares

def build_visibility_matrix(camera_ids, all_feature_points):
    """
    Constructs a visibility matrix V of shape (num_cameras, num_points).
    V[i, f_idx] = 1 if feature f_idx is seen by camera i (in camera_ids[i]),
    otherwise 0.
    """
    num_cameras = len(camera_ids)
    num_points = len(all_feature_points)
    
    # Map camera ID -> row index
    cam_index_map = { cam_id: idx for idx, cam_id in enumerate(camera_ids) }
    
    V = np.zeros((num_cameras, num_points), dtype=np.uint8)
    
    for f_idx, feature_dict in enumerate(all_feature_points):
        for cam_id in feature_dict.keys():
            if cam_id in cam_index_map:
                i = cam_index_map[cam_id]
                V[i, f_idx] = 1
    
    return V

def pack_params(camera_params, points_3d):
    """
    camera_params: list of shape (num_cameras, 6) for (Rodrigues, translation)
    points_3d   : array of shape (num_points, 3)
    Returns a 1D array combining all camera parameters + 3D points.
    """
    return np.hstack([
        camera_params.ravel(),
        points_3d.ravel()
    ])

def unpack_params(params, num_cameras, num_points):
    """
    Inverse of pack_params:
    params      : 1D array of length num_cameras*6 + num_points*3
    num_cameras : number of cameras
    num_points  : number of 3D points
    Returns: (camera_params, points_3d)
    """
    camera_params_size = num_cameras * 6
    cameras = params[:camera_params_size].reshape((num_cameras, 6))
    points  = params[camera_params_size:].reshape((num_points, 3))
    return cameras, points



def reprojection_residual(params, num_cameras, num_points, K, V, points_2d):
    """
    params    : 1D array of length (num_cameras*6 + num_points*3)
    num_cameras : number of cameras
    num_points  : number of 3D points
    K         : 3x3 camera intrinsic matrix
    V         : visibility matrix, shape (num_cameras, num_points)
    points_2d : list (or array) of shape (num_cameras, num_points, 2)
                or -1 placeholders for invisible points

    Returns a 1D residual vector giving the reprojection error for all visible points
    in all cameras.
    """
    camera_params, points_3d = unpack_params(params, num_cameras, num_points)
    
    residuals = []
    
    for i in range(num_cameras):
        # Extract camera i's parameters
        rvec = camera_params[i, :3]   # Rodrigues
        tvec = camera_params[i, 3:]   # translation
        
        # Convert Rodrigues -> Rotation matrix
        Rmat, _ = cv2.Rodrigues(rvec)
        
        for j in range(num_points):
            if V[i, j] == 0:
                # Not visible, skip
                continue
            
            # 2D observation
            u_obs, v_obs = points_2d[i, j]
            
            # 3D point
            X, Y, Z = points_3d[j]
            
            # Reproject
            # World -> camera coords
            XYZ_cam = Rmat @ np.array([X, Y, Z]) + tvec
            
            # guard if behind camera (Z <= 0)
            if XYZ_cam[2] <= 1e-8:
                # This might cause problems in real BA code if a point is behind the camera,
                # but for simplicity, let's just skip or add a large penalty
                continue
            
            x_proj = XYZ_cam[0] / XYZ_cam[2]
            y_proj = XYZ_cam[1] / XYZ_cam[2]
            
            # Apply intrinsics K
            # [fx 0 cx]
            # [0 fy cy]
            # [0  0  1 ]
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            
            u_proj = fx*x_proj + cx
            v_proj = fy*y_proj + cy
            
            # Reprojection error
            residuals.append(u_proj - u_obs)
            residuals.append(v_proj - v_obs)
    
    return np.array(residuals, dtype=np.float64)


def bundle_adjustment(
    K,
    camera_params_init,  # shape (num_cameras, 6)
    points_3d_init,      # shape (num_points, 3)
    points_2d,           # shape (num_cameras, num_points, 2)
    visibility_matrix    # shape (num_cameras, num_points), 1 if visible, 0 otherwise
):
    """
    Runs a simple bundle adjustment, returning refined camera_params and points_3d.
    """
    num_cameras = camera_params_init.shape[0]
    num_points = points_3d_init.shape[0]
    
    print(num_cameras, num_points, points_2d.shape)
    
    # Pack initial guess
    x0 = pack_params(camera_params_init, points_3d_init)
    
    # Define our objective function for least_squares
    def fun(params):
        return reprojection_residual(
            params, num_cameras, num_points, K, visibility_matrix, points_2d
        )
    
    # Run the solver
    result = least_squares(
        fun,
        x0,
        method='trf',     # Levenberg–Marquardt (or 'trf' or 'dogbox')
        max_nfev=10
    )
    
    # Unpack refined parameters
    cam_params_opt, points_3d_opt = unpack_params(
        result.x,
        num_cameras,
        num_points
    )
    return cam_params_opt, points_3d_opt, result.cost, result.success
