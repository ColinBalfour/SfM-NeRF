import numpy as np
from Utils import *
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def LinearPnP(X3d,x2d,K):
    N = X3d.shape[0]
    #A = np.zeros((2*N, 12))
    A = None
    K_inv = np.linalg.inv(K)
    #Normalization first
    x_normalized = np.zeros((N, 2))

    for i in range(N):
        p = np.dot(K_inv, np.array([x2d[i][0], x2d[i][1], 1.0]))
        x_normalized[i] = p[:2]


    for i in range(N):
        X, Y, Z = X3d[i]
        #x, y = x2d[i]
        x, y = x_normalized[i]


        # fill A matrix
        A_1 = [X, Y, Z, 1, 0,0, 0, 0 , -x*X, -x*Y, -x*Z, -x]
        A_2 = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]

        A_rows = np.array([A_1, A_2])

        # Stack onto the existing A matrix
        if A is None:
            A = A_rows
        else:
            A = np.vstack((A, A_rows))

    # Solve for P using SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    R_est = P[:, :3]

    #P_123 = P[:, :3]

    # Apply K⁻¹ to get R̂ = K⁻¹[p₁ p₂ p₃]
    #K_inv = np.linalg.inv(K)
    #R_est = np.dot(K_inv, P_123) # This is K⁻¹[P₁ P₂ P₃]

    # SVD cleanup
    U, D, Vt = np.linalg.svd(R_est)
    R = np.dot(U, Vt)  # enforce orthonormality

    if np.linalg.det(R) < 0:
        R = -R

    # # Translation
    p4 = P[:, 3]

    #t = np.dot(K_inv, p4)
    t = p4

    T = p4/D[0] # translation after scale recovery

    # Camera Centre
    C = -np.dot(R.T, T) # t should be after scale

    return C, R


# def LinearPnP_CrossProduct(X3d, x2d, K):
#     N = X3d.shape[0]
#
#     # Normalize image points with K
#     K_inv = np.linalg.inv(K)
#     x_normalized = np.zeros((N, 3))
#
#     for i in range(N):
#         x_normalized[i] = K_inv @ np.array([x2d[i][0], x2d[i][1], 1])
#
#     # Build matrix A using cross product formulation
#     A = np.zeros((3 * N, 12))
#
#     for i in range(N):
#         X, Y, Z = X3d[i]
#         u, v, w = x_normalized[i]  # w should be ~1 after normalization
#
#         # Create skew-symmetric matrix for cross product
#         skew_x = np.array([
#             [0, -w, v],
#             [w, 0, -u],
#             [-v, u, 0]
#         ])
#
#         # Create X_tilde (expanded X for all rows of P)
#         X_homo = np.array([X, Y, Z, 1])
#         X_tilde = np.zeros((3, 12))
#
#         # Fill X_tilde with the 3D point in the right positions
#         X_tilde[0, 0:4] = X_homo
#         X_tilde[1, 4:8] = X_homo
#         X_tilde[2, 8:12] = X_homo
#
#         # Compute skew_x × X_tilde and store in A
#         A[3 * i:3 * (i + 1), :] = skew_x @ X_tilde
#
#     # Solve using SVD
#     _, _, Vt = np.linalg.svd(A)
#     P = Vt[-1].reshape(3, 4)
#
#     # Extract rotation and translation
#     R_est = P[:, :3]
#     t = P[:, 3]
#
#     # Use SVD to enforce orthogonality of R
#     U, D, Vt = np.linalg.svd(R_est)
#     R = U @ Vt
#
#     # Check determinant and adjust if needed
#     if np.linalg.det(R) < 0:
#         R = -R
#         t = -t
#
#     # Calculate camera center
#     C = -R.T @ t
#
#     # Scale recovery
#     # Note: The scale factor should be the first singular value
#     scale = D[0]
#     t_scale = t / scale
#
#     return R, t, C, t_scale

def reprojectionErrorPnP(X, x, K, R, C):
    """
    Calculate reprojection error for PnP.

    Parameters:
    X: 3D points (N x 3)
    x: 2D points (N x 2)
    K: Camera intrinsic matrix (3 x 3)
    R: Rotation matrix (3 x 3)
    C: Camera center (3 x 1)

    Returns:
    Mean reprojection error in pixels
    """
    # Convert camera center to translation vector
    T = -np.dot(R, C.reshape(3, 1))

    # Project 3D points to image
    total_error = 0
    for i in range(len(X)):
        X_i = X[i]
        x_i = x[i]
        proj_x, proj_y = project_point_to_image(X_i, R, T, K)
        error = np.sqrt((proj_x - x_i[0]) ** 2 + (proj_y - x_i[1]) ** 2)
        total_error += error

    return total_error / len(X)

def PnPRANSAC(X3d, x2d, K, num_iter=10000, threshold=10.0):
    N = X3d.shape[0]

    best_inliers = []
    best_R = None
    best_T = None

    for i in range(num_iter):
        #Randomly select 6 correspondences
        indices = np.random.choice(N, 6, replace=False)
        X_sample = X3d[indices]
        x_sample = x2d[indices]

        try:
            #Compute camera pose from sample
            C, R = LinearPnP(X_sample, x_sample, K)
            C = C.reshape(3,1)
            #reprojection errors for all points
            inliers = []
            for j in range(N):
                # Project 3D point to image
               # X = np.append(X3d[j], 1)  # Homogeneous coordinates
                error, _, _ = reprojection_error(X3d[j], x2d[j], R, C, K)
                if error < threshold:
                    inliers.append(j)

            # Update best model if we found more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_R = R
                best_T = C
        except:
            print("Error")
            continue


    return best_R, best_T, best_inliers

def NonlinearPnP(X3d, x2d, K, R_init, C_init):
    # Convert rotation matrix to quaternion for optimization
    r = Rotation.from_matrix(R_init)
    quat_init = r.as_quat()  # [x, y, z, w] format
    params_init = np.concatenate([quat_init, C_init.flatten()])
    I = np.identity(3)
    def residuals(params):
        # Extract quaternion and camera center from parameters
        quat = params[:4]
        C = params[4:].reshape(3, 1)

        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)

        # Convert quaternion to rotation matrix
        r = Rotation.from_quat(quat)
        R = r.as_matrix()

        # Calculate reprojection errors
        errors = []
        geoerrors = []
        for i in range(len(X3d)):
            # Project 3D point
            geo_error, error_x, error_y = reprojection_error(X3d[i], x2d[i], R, C, K)
            errors.extend(error_x)
            errors.extend(error_y)
            geoerrors.extend([geo_error])

        return errors

    # optimization
    result = least_squares(residuals, params_init, method='lm')
    optimized_params = result.x

    # optimized quaternion and camera center
    quat_opt = optimized_params[:4]
    quat_opt = quat_opt / np.linalg.norm(quat_opt)
    C_opt = optimized_params[4:].reshape(3, 1)

    # Convert quaternion back to rotation matrix
    r_opt = Rotation.from_quat(quat_opt)
    R_opt = r_opt.as_matrix()

    # Calculate final mean reprojection error
    final_errors = residuals(optimized_params)
    rms_error = np.sqrt(np.mean(np.square(final_errors)))
    print(f"Non-linear refinement: RMS reprojection error = {rms_error:.4f} pixels")

    return  C_opt, R_opt