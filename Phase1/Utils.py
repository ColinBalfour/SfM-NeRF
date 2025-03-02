import numpy as np

def project_point_to_image(X, R, T, K):
    """
    Project a 3D point to image coordinates.

    Parameters:
    X: 3D point (3,) or (3,1)
    R: Rotation matrix (3,3)
    T: Translation vector (3,1)
    K: Camera intrinsic matrix (3,3)

    Returns:
    x, y: image coordinates
    """
    # Ensure X is shape (3,)
    X = X.flatten()[:3]

    # Make homogeneous 3D point
    X_homo = np.append(X, 1)

    # Construct projection matrix
    P = np.dot(K, np.hstack((R, T)))

    # Project point
    x_proj_homo = np.dot(P, X_homo)

    # Convert from homogeneous to image coordinates
    x_proj = x_proj_homo[:2] / x_proj_homo[2]

    return x_proj[0], x_proj[1]


def projectedpointframe(R, T, R_final, C_final, K, X_final):
    """
    Project points to two camera frames.

    Parameters:
    R: Rotation matrix of first camera
    T: Translation vector of first camera
    R_final: Rotation matrix of second camera
    C_final: Camera center of second camera
    K: Camera intrinsic matrix
    X_final: 3D points to project

    Returns:
    projected_pts_frame1: Points projected to first frame
    projected_pts_frame2: Points projected to second frame
    """
    # Projecting points back to images after linear triangulation
    # For frame 1 (reference frame)
    R1 = R
    T1 = T
    T2 = -np.dot(R_final, C_final.reshape(3, 1))

    # frame 1
    projected_pts_frame1 = []
    for point in X_final:
        x, y = project_point_to_image(point, R1, T1, K)
        projected_pts_frame1.append((x, y))

    # For frame 2
    projected_pts_frame2 = []
    for point in X_final:
        x, y = project_point_to_image(point, R_final, T2, K)
        projected_pts_frame2.append((x, y))

    return np.array(projected_pts_frame1), np.array(projected_pts_frame2)

def mean_reprojection_error(fpts1, fpts2, X_final, R, T, R_final, C_final, K):
    # Reprojection error after linear triangulation
    total_error_1 = []  # frame 1
    total_error_2 = []  # frame 2
    for pts1, pts2, X in zip(fpts1, fpts2, X_final):
        error_1, _, _ = reprojection_error(X, pts1, R, T, K)
       # print(f"Error for pts1{error_1}")
        total_error_1.append(error_1)
        error_2, _, _ = reprojection_error(X, pts2, R_final, C_final, K)
        #print(f"Error for pts2{error_2}")
        total_error_2.append(error_2)
    mean_error_1 = np.mean(total_error_1)
    mean_error_2 = np.mean(total_error_2)
    Mean_Error = (mean_error_1 + mean_error_2) / 2.0
    return  mean_error_1, mean_error_2, Mean_Error

def reprojection_error(X, x, R, C, K ):
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    # rows of projection matrices P2
    P1 = P[0, :].reshape(1, 4)
    P2 = P[1, :].reshape(1, 4)
    P3 = P[2, :].reshape(1, 4)

    u, v = x

    # Convert X to homogeneous coordinates
    X = np.append(X, 1) if X.shape[0] == 3 else X

    a = (np.dot(P1, X) / np.dot(P3, X))
    b = (np.dot(P2, X) / np.dot(P3, X))
    error_x = u - a
    error_y = v - b

    error = (u - a) ** 2 + (v - b) ** 2

    return error, error_x, error_y

def get_skew_mat(a):
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])
