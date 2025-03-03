import numpy as np
from scipy.optimize import least_squares
from Utils import *

# def linear_triangulation(K, R, T, pose, final_pts1, final_pts2):
#     I = np.identity(3)
#     P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
#     P2 = []
#     X = []
#     X_all = []
#     X_best = []
#     best_R = []
#     best_T = []
#     for (C,R2) in pose:
#         C = C.reshape(3, 1)
#         P = np.dot(K, np.dot(R2, np.hstack((I, -C))))
#         for i, (p1,p2) in enumerate(zip(final_pts1, final_pts2)):
#             x1, y1 = p1
#             x2, y2 = p2
#             xm  = np.array([
#                     [0, -1, y1],
#                     [1, 0, -x1],
#                     [-y1, x1, 0]
#                 ])
#             xcm = np.array([
#                 [0, -1, y2],
#                 [1, 0, -x2],
#                 [-y2, x2, 0]
#             ])
#             A = np.vstack((
#                 np.dot(xm, P1),
#                 np.dot(xcm, P)
#             ))
#
#             # Solve for X using SVD
#             _, _, Vt = np.linalg.svd(A)
#             X_1= Vt[-1]
#
#             X_3d = X_1[:3] / X_1[3]
#             r3 = R2[:, 2]
#             X.append(X_3d)
#             if np.dot(r3, (X_3d - C.flatten())) > 0:
#                 X_best.append(X_3d)
#                 best_R.append(R2)
#                 best_T.append(C)
#         X_all.append(X)
#
#     return X_best, best_R, best_T, X_all

# def get_skew_mat(a):
#     return np.array([
#         [0, -a[2], a[1]],
#         [a[2], 0, -a[0]],
#         [-a[1], a[0], 0]
#     ])
#
#
# def linear_triangulation(K, R, T, pose, final_pts1, final_pts2):
#     I = np.identity(3)
#     P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
#     X_all = [] # List to store points for all four solutions
#
#     best_pose = 0
#
#     for (C, R2) in pose:
#         X_current = []  # Points for current solution
#         valid_point = []
#         vp = {}
#         positive_Z = 0
#         C = C.reshape(3, 1)
#         P2 = np.dot(K, np.dot(R2, np.hstack((I, -C))))
#
#         for i, (p1, p2) in enumerate(zip(final_pts1, final_pts2)):
#             x1, y1 = p1
#             x2, y2 = p2
#
#             p1 = np.array([x1, y1, 1])
#             p2 = np.array([x2, y2, 1])
#
#             # Construct linear system A for current point
#             A = np.vstack([
#                 get_skew_mat(p1) @ P1,
#                 get_skew_mat(p2) @ P2
#             ])
#
#             # Solve using SVD
#             _, _, Vt = np.linalg.svd(A)
#             X = Vt[-1]  # Take the last row of Vt
#             X_3d = X[:3] / X[3]
#             # Store point for current solution
#             X_current.append(X_3d)
#             v = X_3d - C.flatten()
#
#             # Check cheirality condition
#             r3 = R2[2, :]
#             print("r3_shape", r3.shape)
#             print("v", v.shape)
#             print("dot product", np.dot(r3, v) )
#             if np.dot(r3, v) > 0 and X_3d[2] > 0:
#                 positive_Z += 1
#                 #valid_point.append(X_3d)
#                 #vp['j'] = X_3d
#
#         # Add all points for current solution to X_all
#         X_all.append(X_current)
#         if positive_Z > best_pose:
#             best_pose = positive_Z
#             #X_best = valid_point
#             X_best = X_current
#             best_R = R2
#             best_T = C
#
#     # Convert lists to numpy arrays
#     X_best = np.array(X_best)
#     X_all = np.array(X_all)
#     print("X_best", valid_point)
#     print("mapping", vp)
#
#     return X_best, best_R, best_T, X_all



# def linear_triangulation(K, R, T, pose, final_pts1, final_pts2):
#     I = np.identity(3)
#     P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
#     X_triangulated = []
#     for (C, R2) in pose:
#         print("Camera Centre and Rotation", (C, R2))
#         X_current = []  # Points for current solution
#         C = C.reshape(3, 1)
#         P2 = np.dot(K, np.dot(R2, np.hstack((I, -C))))
#         for i, (p1, p2) in enumerate(zip(final_pts1, final_pts2)):
#             x1, y1 = p1
#             x2, y2 = p2
#
#             p1 = np.array([x1, y1, 1])
#             p2 = np.array([x2, y2, 1])
#
#             # Construct linear system A for current point
#             A = np.vstack([
#                 get_skew_mat(p1) @ P1,
#                 get_skew_mat(p2) @ P2
#             ])
#
#             # Solve using SVD
#             _, _, Vt = np.linalg.svd(A)
#             X = Vt[-1]  # Take the last row of Vt
#             X_3d = X[:3] / X[3]
#             # Store point for current solution
#             X_current.append(X_3d)
#         print("len(X_current)", len(X_current))
#         X_triangulated.append(X_current)
#         print("X_triangulated:", X_triangulated)
#
#     # checking chirality condition
#     positive_depths = []
#     for i, ((C, R) , X3d) in enumerate(zip(pose, X_triangulated)):
#         depth = 0
#         for j in range(len(X3d)):
#             X = X3d[j].reshape(-1,1)
#             r3 = R[2, :]
#             #print("shape of r3", r3.shape)
#             v = X - C.reshape(-1,1)
#             #print("shape of dot product of r3 and v", np.dot(r3, v))
#             #print("shape of v", v.shape)
#             if np.dot(r3, v) > 0.1 and X[2] > 0.1:
#                 depth = depth + 1
#         positive_depths.append(depth)
#     print("positive_depths:", positive_depths)
#     max_depth = max(positive_depths)
#     max_index = positive_depths.index(max_depth)
#     X_best = X_triangulated[max_index]
#     print(f" We found best Centre and Rotation matrix is pose{max_index+1}: {pose[max_index]}")
#     # returning camera centre and rotation
#
#     return np.array(X_best), pose[max_index], X_triangulated


# def triangulate_points_cv(K, R1, T1, R2, T2, pts1, pts2):
#     """
#     Triangulate points using OpenCV's triangulatePoints function.
#
#     Parameters:
#     K: Camera calibration matrix (3x3)
#     R1, T1: Rotation and translation of first camera
#     R2, T2: Rotation and translation of second camera
#     pts1, pts2: Matched points in the two images
#
#     Returns:
#     points_3d: Array of triangulated 3D points
#     """
#     # Compute projection matrices for both cameras
#     P1 = np.dot(K, np.hstack((R1, T1)))
#     P2 = np.dot(K, np.hstack((R2, T2)))
#
#     # Convert points to correct format
#     pts1 = np.array(pts1).T
#     pts2 = np.array(pts2).T
#
#     # Triangulate points
#     points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
#
#     # Convert from homogeneous coordinates to 3D
#     points_3d = points_4d[:3] / points_4d[3]
#
#     return points_3d.T  # Transpose to get points as rows

def triangulationlinear(K, R1, T1, R2, T2, fpts1,fpts2):
    """


    Parameters:
    K: Camera calibration matrix (3x3)
    R1, T1: Rotation and translation of first camera
    R2, T2: Rotation and translation of second camera
    pts1, pts2: Matched points in the two images

    Returns:
    points_3d: Array of triangulated 3D points
    """
    # Compute projection matrices for both cameras
    P1 = np.dot(K, np.hstack((R1, T1)))
    P2 = np.dot(K, np.hstack((R2, T2)))

    # Normalize coordinates (optional but can improve stability)
    K_inv = np.linalg.inv(K)
    X_wp = []
    for pts1, pts2 in zip(fpts1, fpts2):
        x1, y1 = pts1
        x2, y2 = pts2
        p1  = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])

        # Optional: normalize coordinates
        p1_norm = K_inv @ p1
        p2_norm = K_inv @ p2
        # p1 = p1_norm
        # p2 = p2_norm

        pts1_cross = get_skew_mat(p1)
        pts2_cross = get_skew_mat(p2)

        A_1 = pts1_cross @ P1
        A_2 = pts2_cross @ P2
        A = np.vstack((A_1, A_2))

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Take the last row of Vt
        X_3d = X[:3] / X[3]
        X_wp.append(X_3d)
    return X_wp # Transpose to get points as rows


def optimization(K, R1, T1, R2, T2, pts1, pts2, X_i):
    def objective(X):
        return residuals(X, K, R1, T1, R2, T2, pts1, pts2)

    result = least_squares(
        objective,
        X_i,
        method='trf',  # Trust Region Reflective algorithm
        loss='linear',  # Standard least squares # Maximum number of function evaluations
        verbose=0  # No printing
    )

    return result.x


def residuals(X_1, K, R, T, R2, T2, pts1, pts2):
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -T2))))

    # rows of projection matrices P1
    P1_1 = P1[0, :].reshape(1, 4)
    P1_2 = P1[1, :].reshape(1, 4)
    P1_3 = P1[2, :].reshape(1, 4)

    # rows of projection matrices P2
    P2_1 = P2[0, :].reshape(1, 4)
    P2_2 = P2[1, :].reshape(1, 4)
    P2_3 = P2[2, :].reshape(1, 4)

    x1, y1 = pts1
    x2, y2 = pts2
    # Convert X to homogeneous coordinates if not already
    X_1 = np.append(X_1, 1) if X_1.shape[0] == 3 else X_1

    a = (np.dot(P1_1, X_1) / np.dot(P1_3, X_1))
    b = (np.dot(P1_2, X_1) / np.dot(P1_3, X_1))
    c = (np.dot(P2_1, X_1) / np.dot(P2_3, X_1))
    d = (np.dot(P2_2, X_1) / np.dot(P2_3, X_1))

    # geo_error_1 = (a - x1) ** 2 + (b - y1) ** 2
    # geo_error_2 = (c - x2) ** 2 + (d - y2) ** 2
    # error = geo_error_1 + geo_error_2

    error_x1 = a - x1
    error_y1 = b - y1
    error_x2 = c - x2
    error_y2 = d - y2

    return np.array([error_x1, error_y1, error_x2, error_y2]).flatten()


def non_linear_triangulation(K, R1, T1, R2, T2, finalpts1, finalpts2, X):
    X_optimized = []
    initial_errors = []
    final_errors = []

    for pts1, pts2, X_i in zip(finalpts1, finalpts2, X):
        # Initial error
        initial_residuals = residuals(X_i[:3], K, R1, T1, R2, T2, pts1, pts2)
        initial_error = np.sum(initial_residuals ** 2)  # Sum of squared residuals
        initial_errors.append(initial_error)

        # Optimize
        X_opt = optimization(K, R1, T1, R2, T2, pts1, pts2, X_i)
        X_optimized.append(X_opt)

        # Final error
        final_residuals = residuals(X_opt, K, R1, T1, R2, T2, pts1, pts2)
        final_error = np.sum(final_residuals ** 2)  # Sum of squared residuals
        final_errors.append(final_error)

    print(f"Mean initial error: {np.mean(initial_errors):.6f}")
    print(f"Mean final error: {np.mean(final_errors):.6f}")
    print(f"Error reduction: {100 * (1 - np.mean(final_errors) / np.mean(initial_errors)):.2f}%")

    return np.array(X_optimized)


# def non_linear_triangulation(K, R, T, R2, T2, final_pts1, final_pts2, X):
#     I = np.identity(3)
#     P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
#     P2 = np.dot(K, np.dot(R2, np.hstack((I, -T2))))
#
#     # rows of projection matrices P1
#     P1_1 = P1[0, :].reshape(1, 4)
#     P1_2 = P1[1, :].reshape(1, 4)
#     P1_3 = P1[2, :].reshape(1, 4)
#
#     # rows of projection matrices P2
#     P2_1 = P2[0, :].reshape(1, 4)
#     P2_2 = P2[1, :].reshape(1, 4)
#     P2_3 = P2[2, :].reshape(1, 4)
#
#     # error
#     error = []
#
#     for (p1, p2, X_1) in zip(final_pts1, final_pts2, X):
#         x1, y1 = p1
#         x2, y2 = p2
#         # Convert X to homogeneous coordinates if not already
#         X_1 = np.append(X_1, 1) if X_1.shape[0] == 3 else X_1
#
#         a = (np.dot(P1_1, X_1)/np.dot(P1_3, X_1))
#         b = (np.dot(P1_2, X_1)/np.dot(P1_3, X_1))
#         c = (np.dot(P2_1, X_1)/np.dot(P2_3, X_1))
#         d = (np.dot(P2_2, X_1)/np.dot(P2_3, X_1))
#         geo_error_1 = np.sum((a - x1)**2 + (b - y1)**2)
#         geo_error_2 = np.sum((c - x2)**2 + (d - y2)**2)
#
#        # error = np.vstack((geo_error_1, geo_error_2))
#         point_error = np.array([geo_error_1, geo_error_2])
#         error.append(point_error)
#
#     return


def chirality_condition(triangulated_points, pose):
    
    # Now check which pose gives the most points in front of both cameras
    best_pose_idx = 0
    max_valid_points = 0

    for i, (points, (C, R)) in enumerate(zip(triangulated_points, pose)):
        # Check cheirality condition
        valid_points = 0
        for pt in points:
            z1 = pt[2]
            if z1 > 0:
                # Check if point is in front of second camera
                r3 = R[2, :]
                v = pt - C.reshape(3)
                if np.dot(r3, v) > 0:
                    valid_points += 1
        print(f"Pose {i + 1}: {valid_points} valid points")
        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_pose_idx = i
    
    print(f"Best pose: {best_pose_idx}")

    # Get the best pose and points
    X_final = np.array(triangulated_points[best_pose_idx])
    C_final = pose[best_pose_idx][0].reshape(3,1)
    R_final = pose[best_pose_idx][1]

    return X_final, C_final, R_final