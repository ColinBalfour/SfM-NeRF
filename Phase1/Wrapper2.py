import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import least_squares
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import numpy.linalg


def load_calibration(calib_file):
    """
    Load the camera intrinsic parameters (K) from calibration.txt.
    Assumes calibration.txt contains 3 rows with 3 numbers each.
    """
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    K = []
    for line in lines:
        line = line.strip()
        if line:
            values = list(map(float, line.split()))
            K.append(values)
    K = np.array(K)
    return K


def load_images(path, num_imgs):
    """
    Load images from the given folder.
    Assumes image filenames are "1.png", "2.png", ... up to num_imgs.
    """
    images = []
    for i in range(1, num_imgs + 1):
        img_path = os.path.join(path, f"{i}.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
        else:
            images.append(img)
    return images


def parse_matching_file(filename):
    """
    Parse a matching file.

    File format:
      - The first line is of the form "nFeatures: <number>"
      - Each subsequent line corresponds to one feature.
        Format per line:
          total_imgs R G B u_curr v_curr [img_id u_match v_match]...
    """
    matches = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header: e.g., "nFeatures: 3930"
    header = lines[0].strip()
    if not header.startswith("nFeatures:"):
        raise ValueError("Invalid matching file format; missing 'nFeatures:' header.")
    n_features = int(header.split(":")[1])

    # Parse each feature line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        total_imgs = int(tokens[0])
        color = tuple(map(int, tokens[1:4]))
        pt_curr = tuple(map(float, tokens[4:6]))

        num_matches = total_imgs - 1
        match_list = []
        for i in range(num_matches):
            offset = 6 + i * 3
            img_id = int(tokens[offset])
            u = float(tokens[offset + 1])
            v = float(tokens[offset + 2])
            match_list.append({'image_id': img_id, 'pt': (u, v)})

        matches.append({
            'color': color,
            'pt_curr': pt_curr,
            'matches': match_list
        })

    return n_features, matches


def get_keypoints_and_matches_for_pair(matches, target_img_id):
    """
    Given a list of features (from matching file for image1),
    extract keypoints and matching points corresponding to a target image id.
    Returns:
        kp1: list of cv2.KeyPoint for the current (source) image.
        kp2: list of cv2.KeyPoint for the target image.
        dmatches: list of cv2.DMatch linking kp1 to kp2.
    """
    kp1 = []
    kp2 = []
    dmatches = []

    for feature in matches:
        # Check all matches for the current feature for the target image id.
        for m in feature['matches']:
            if m['image_id'] == target_img_id:
                pt1 = feature['pt_curr']
                pt2 = m['pt']
                # Create KeyPoint objects (using an arbitrary size, e.g., 5)
                keypoint1 = cv2.KeyPoint(x=pt1[0], y=pt1[1], size=5)
                keypoint2 = cv2.KeyPoint(x=pt2[0], y=pt2[1], size=5)
                kp1.append(keypoint1)
                kp2.append(keypoint2)
                # Create a DMatch object. The indices correspond to the order in the lists.
                match = cv2.DMatch(_queryIdx=len(kp1) - 1, _trainIdx=len(kp2) - 1, _distance=0)
                dmatches.append(match)
                break  # Only use the first match found for this feature.

    return kp1, kp2, dmatches


def display_matches(img1, img2, kp1, kp2, dmatches):
    """
    Draws and displays the matches between two images using cv2.drawMatches.
    """
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, dmatches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Feature Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def estimate_fundamental_matrix(pts1, pts2):
    """
    Estimate the fundamental matrix from the given keypoints and matches.
    """
    # Get with OpenCV
    # F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    # print("opencv:")
    # print(F)
    # print(np.linalg.matrix_rank(F))

    # Estimate the fundamental matrix
    A = np.zeros((len(pts1), 9))
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        x1, y1 = p1
        x2, y2 = p2
        A[i, :] = np.array([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    # Solve for F using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set smallest singular value to zero
    F = U @ np.diag(S) @ Vt  # Reconstruct F with rank-2 constraint

    F = F / F[2, 2]  # enforce F(3,3) = 1

    # print("custom:")
    # print(F)
    # print(np.linalg.matrix_rank(F))

    return F


def reject_outliers(kp1, kp2, dmatches, N=15000, threshold=.005):
    """
    Reject outliers using RANSAC.
    """
    # Extract point correspondences
    pts1 = np.array([kp1[m.queryIdx].pt for m in dmatches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in dmatches])

    # Get with OpenCV
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold)

    print("opencv:")
    print(F)
    print(np.linalg.matrix_rank(F))

    best_F = None
    best_inliers = []
    final_pts1 = []
    final_pts2 = []

    # Use RANSAC to estimate the fundamental matrix
    for i in range(N):
        # Randomly select 8 points
        indices = np.random.choice(len(pts1), 8, replace=False)
        pts1_sample = pts1[indices]
        pts2_sample = pts2[indices]

        # Estimate F using the 8-point algorithm
        F = estimate_fundamental_matrix(pts1_sample, pts2_sample)

        # Count inliers
        inliers = []
        for j in range(len(pts1)):
            x1 = np.append(pts1[j], 1)
            x2 = np.append(pts2[j], 1)
            if np.abs(x1.T @ F @ x2) < threshold:
                inliers.append(j)

        # Update best F if this iteration has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F

    print(f"RANSAC: Found {len(best_inliers)} inliers out of {len(pts1)} matches.")

    recomputed_F = estimate_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    return recomputed_F, best_inliers


def get_essential_mtx(K, F):
    """
    Compute the essential matrix from the fundamental matrix and camera intrinsics.
    """

    E = K.T @ F @ K
    return E


def get_camera_pose(E):
    U, D, Vt = np.linalg.svd(E)
    print("D", D)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    # Computing all possible pairs
    # Here C is Camera Centre
    C1 = U[:, 2]
    C2 = -U[:, 2]

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    pose = [(C1, R1), (C2, R1), (C1, R2), (C2, R2)]
    correct_pose = []
    for C, R in pose:
        if np.linalg.det(R) < 0:
            C = -C
            R = -R
        correct_pose.append((C, R))
    return correct_pose


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


def linear_triangulation(K, R, T, pose, final_pts1, final_pts2):
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
    X_triangulated = []
    best_pose = 0
    for (C, R2) in pose:
        print("Camera Centre and Rotation", (C, R2))
        X_current = []  # Points for current solution
        C = C.reshape(3, 1)
        P2 = np.dot(K, np.dot(R2, np.hstack((I, -C))))
        for i, (p1, p2) in enumerate(zip(final_pts1, final_pts2)):
            x1, y1 = p1
            x2, y2 = p2

            p1 = np.array([x1, y1, 1])
            p2 = np.array([x2, y2, 1])

            # Construct linear system A for current point
            A = np.vstack([
                get_skew_mat(p1) @ P1,
                get_skew_mat(p2) @ P2
            ])

            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]  # Take the last row of Vt
            X_3d = X[:3] / X[3]
            # Store point for current solution
            X_current.append(X_3d)
        print("len(X_current)", len(X_current))
        X_triangulated.append(X_current)
        print("X_triangulated:", X_triangulated)

    # checking chirality condition
    positive_depths = []
    for i, ((C, R) , X3d) in enumerate(zip(pose, X_triangulated)):
        depth = 0
        for j in range(len(X3d)):
            X = X3d[j].reshape(-1,1)
            r3 = R[2, :]
            #print("shape of r3", r3.shape)
            v = X - C.reshape(-1,1)
            #print("shape of dot product of r3 and v", np.dot(r3, v))
            #print("shape of v", v.shape)
            if np.dot(r3, v) > 0 and X[2] > 0:
                depth = depth + 1
        positive_depths.append(depth)
    print("positive_depths:", positive_depths)
    max_depth = max(positive_depths)
    max_index = positive_depths.index(max_depth)
    X_best = X_triangulated[max_index]
    print(f"best centre and rotation matrix of camera pose{max_index+1}: {pose[max_index]}")

    return np.array(X_best), pose[max_index]


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

    geo_error_1 = (a - x1) ** 2 + (b - y1) ** 2
    geo_error_2 = (c - x2) ** 2 + (d - y2) ** 2
    error = geo_error_1 + geo_error_2

    return error


def non_linear_triangulation(K, R1, T1, R2, T2, finalpts1, finalpts2, X):
    X_optimized = []
    initial_errors = []
    final_errors = []
    for pts1, pts2, X_i in zip(finalpts1, finalpts2, X):
        # intial error
        initial_residuals = residuals(X_i[:3], K, R1, T1, R2, T2, pts1, pts2)
        initial_error = np.sum(initial_residuals ** 2)
        initial_errors.append(initial_error)

        X_opt = optimization(K, R1, T1, R2, T2, pts1, pts2, X_i)
        X_optimized.append(X_opt)

        # final error
        final_residuals = residuals(X_opt, K, R1, T1, R2, T2, pts1, pts2)
        final_error = np.sum(final_residuals ** 2)
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


def PnPRANSAC(X3d, x2d, K, num_iter=1000, threshold=2.0):
    N = X3d.shape[0]
    I = np.identity(3)
    print("N",N)
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
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -T))))

    # Project point
    x_proj_homo = np.dot(P, X_homo)

    # Convert from homogeneous to image coordinates
    x_proj = x_proj_homo[:2] / x_proj_homo[2]

    return x_proj[0], x_proj[1]


def plot_dual_view_triangulation(points_data, x_min=None, x_max=None, z_min=None, z_max=None):
    """
    Create side-by-side plots showing X vs Y (front view) and X vs Z (top view)
    of world points.

    Parameters:
    points_data: Either a single array of 3D points or a list of arrays of 3D points
    x_min, x_max: Optional limits for X axis
    z_min, z_max: Optional limits for Z axis
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ensure points_data is a list
    if not isinstance(points_data, list):
        points_data = [points_data]

    # Colors for different solutions
    colors = ['blue', 'red', 'green', 'cyan']

    # Plot points in both views
    for i, points in enumerate(points_data):
        points = np.array(points)

        # Check if points is empty
        if points.size == 0:
            continue

        # Handle different array shapes
        if len(points.shape) == 1:
            # Try to reshape if it's a 1D array
            if points.size % 3 == 0:
                points = points.reshape(-1, 3)
            else:
                print(f"Warning: Cannot reshape points with shape {points.shape} into Nx3 array")
                continue

        # Now we should have a proper Nx3 array
        try:
            # Front view (X vs Y)
            ax1.scatter(points[:, 0], points[:, 1],  # X vs Y coordinates
                        c=colors[i % len(colors)], s=1, alpha=0.5,
                        label=f'Solution {i + 1}')

            # Top view (X vs Z)
            ax2.scatter(points[:, 0], points[:, 2],  # X vs Z coordinates
                        c=colors[i % len(colors)], s=1, alpha=0.5)
        except IndexError as e:
            print(f"Error plotting points with shape {points.shape}: {e}")
            print("First few elements:", points[:min(5, len(points))])
            continue

    # Set labels and titles
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Front View (X vs Y)')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Top View (X vs Z)')

    # Add grid to both plots
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Set equal aspect ratio for both plots
    ax1.axis('equal')
    ax2.axis('equal')

    # Add legend to the first plot only
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set specific X and Z axis limits if provided
    if x_min is not None and x_max is not None:
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)

    if z_min is not None and z_max is not None:
        ax2.set_ylim(z_min, z_max)  # Z axis is the Y-axis in the second plot

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def main():
    # Set data folder and number of images
    path = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    #path = ""
    num_imgs = 5

    # Load images (which are already undistorted and resized to 800x600px)
    images = load_images(path, num_imgs)
    if len(images) < 2:
        print("Need at least two images to match.")
        return
    path2 = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"

    # Load camera calibration parameters (intrinsic matrix K) if needed
    calib_file = os.path.join(path2, "calibration.txt")
    try:
        K = load_calibration(calib_file)
        print("Camera intrinsic matrix K:")
        print(K)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return

    # Load matching data for image1 from matching1.txt
    matching_file = os.path.join(path2, "matching1.txt")

    try:
        _, matches = parse_matching_file(matching_file)
    except Exception as e:
        print(f"Error parsing matching file: {e}")
        return

    # For demonstration, we display matches between image 1 and image 2.
    target_img_id = 2  # Change this if you want to display matches with a different image.
    kp1, kp2, dmatches = get_keypoints_and_matches_for_pair(matches, target_img_id)
    print(f"Found {len(kp1)} matches between image 1 and image {target_img_id}")

    # Display the matches using cv2.drawMatches
    display_matches(images[0], images[target_img_id - 1], kp1, kp2, dmatches)

    # Estimate the fundamental matrix
    F, inliers = reject_outliers(kp1, kp2, dmatches)

    print("Estimated fundamental matrix F:")
    print(F)

    # Display the inlier matches using cv2.drawMatches
    inlier_matches = [dmatches[i] for i in inliers]
    display_matches(images[0], images[target_img_id - 1], kp1, kp2, inlier_matches)

    fpts1 = np.array([kp1[m.queryIdx].pt for m in inlier_matches])
    fpts2 = np.array([kp2[m.trainIdx].pt for m in inlier_matches])

    E = get_essential_mtx(K, F)
    print("Estimated essential matrix E:")
    print(E)

    pose = get_camera_pose(E)
    print(f"Camera pose {pose}")

    # Linear triangulation
    # Rotation and translation of base frame
    R = np.identity(3)
    T = np.zeros((3, 1))

    X_final, camera_pose = linear_triangulation(K, R, T, pose, fpts1, fpts2)
    R_final = np.array(camera_pose[1]).reshape(3,3)
    T_final = np.array(camera_pose[0]).reshape(3, 1)
    # error_linear_triangulation, _, _ = reprojection_error(
    #     X_final, fpts2, R_final, T_final, K
    # )
    # print("Linear triangulation error:", error_linear_triangulation)

    print("best_R", R_final)
    print("best_T", T_final)

    # Projecting points back to images after linear triangulation
    # For frame 1 (reference frame)
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    projected_pts_frame1 = []

    for point in X_final:
        x, y = project_point_to_image(point, R1, T1, K)
        projected_pts_frame1.append((x, y))

    # For frame 2
    projected_pts_frame2 = []
    for point in X_final:
        x, y = project_point_to_image(point, R_final, T_final, K)
        projected_pts_frame2.append((x, y))

    # Convert to numpy arrays
    projected_pts_frame1 = np.array(projected_pts_frame1)
    projected_pts_frame2 = np.array(projected_pts_frame2)

    # Draw projected points on frame 1
    img1_with_points = images[0].copy()
    for pt in projected_pts_frame1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 1, (0, 255, 0), -1)  # Green circles

    # Draw original matched points on frame 1
    for pt in fpts1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 1, (0, 0, 255), -1)  # Red circles

    # Similarly for frame 2
    img2_with_points = images[1].copy()  # Assuming images[1] is frame 2
    for pt in projected_pts_frame2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 1, (0, 255, 0), -1)  # Green circles

    for pt in fpts2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 1, (0, 0, 255), -1)  # Red circles

    # Display the images
    cv2.imshow("Frame 1 - Green: Projected, Red: Original", img1_with_points)
    cv2.imshow("Frame 2 - Green: Projected, Red: Original", img2_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print("X:", X)
    #print("length of X:", len(X))

    # Nonlinear triangulation
    X_optimized = non_linear_triangulation(K, R, T, R_final, T_final, fpts1, fpts2, X_final)
    X_optimized = np.array(X_optimized)
    print("X_final shape:", X_final.shape)  # Check the shape

    # Projection after non linear triangulation
    # Projecting points back to images after linear triangulation
    # For frame 1 (reference frame)
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    projected_pts_frame1 = []

    for point in X_optimized:
        x, y = project_point_to_image(point, R1, T1, K)
        projected_pts_frame1.append((x, y))

    # For frame 2
    projected_pts_frame2 = []
    for point in X_optimized:
        x, y = project_point_to_image(point, R_final, T_final, K)
        projected_pts_frame2.append((x, y))

    # Convert to numpy arrays
    projected_pts_frame1 = np.array(projected_pts_frame1)
    projected_pts_frame2 = np.array(projected_pts_frame2)

    # Draw projected points on frame 1
    img1_with_points = images[0].copy()
    for pt in projected_pts_frame1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 1, (0, 255, 0), -1)  # Green circles

    # Draw original matched points on frame 1
    for pt in fpts1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 1, (0, 0, 255), -1)  # Red circles

    # Similarly for frame 2
    img2_with_points = images[1].copy()  # Assuming images[1] is frame 2
    for pt in projected_pts_frame2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 1, (0, 255, 0), -1)  # Green circles

    for pt in fpts2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 1, (0, 0, 255), -1)  # Red circles

    # Display the images
    cv2.imshow("Frame 1 after non-linear triangulation - Green: Projected, Red: Original", img1_with_points)
    cv2.imshow("Frame 2  after non-linear triangulation- Green: Projected, Red: Original", img2_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Try reshaping if needed
    if len(X_final.shape) == 1 and X_final.size % 3 == 0:
        X_final_reshaped = X_final.reshape(-1, 3)
        plot_dual_view_triangulation([X_final_reshaped], x_min=-200, x_max=200, z_min=-1000, z_max=1000)
    else:
        # If it's already a 2D array with the right shape
        plot_dual_view_triangulation([X_final], x_min=-200, x_max=200, z_min=-1000, z_max=1000)


    # matching 3rd image
    matching_fil = os.path.join(path2, "matching1.txt")

    try:
        _, matches = parse_matching_file(matching_fil)
    except Exception as e:
        print(f"Error parsing matching file: {e}")
        return
    target_img_id = 3
    kp1, kp3, dmatches3 = get_keypoints_and_matches_for_pair(matches, target_img_id)
    print(f"Found {len(kp1)} matches between image 1 and image {target_img_id}")

    # Display the matches using cv2.drawMatches
    display_matches(images[0], images[target_img_id - 1], kp1, kp3, dmatches3)

    # Estimate the fundamental matrix
    F3, inliers3 = reject_outliers(kp1, kp3, dmatches3)

    print("Estimated fundamental matrix F3:")
    print(F3)

    # Display the inlier matches using cv2.drawMatches
    inlier_matches3 = [dmatches3[i] for i in inliers3]
    display_matches(images[0], images[target_img_id - 1], kp1, kp3, inlier_matches3)

    pts1_img3 = np.array([kp1[m.queryIdx].pt for m in inlier_matches3])
    pts3 = np.array([kp3[m.trainIdx].pt for m in inlier_matches3])

    X_3d = []
    x_2d = []
    threshold = 2.0

    for i, (pt3d, pt1) in enumerate(zip(X_optimized, fpts1)):
        # Find this point in the matching file's image 1 points
        for j, match_pt1 in enumerate(pts1_img3):
            # If we find a very close or exact match
            if np.allclose(pt1, match_pt1, atol=threshold):
                # Add the correspondence:
                # - 3D point from triangulation
                # - 2D point in the target image
                X_3d.append(pt3d)
                x_2d.append(pts3[j])
                break

    # Convert to numpy arrays
    X_3d = np.array(X_3d)
    x_2d = np.array(x_2d)
    print(f"Found {len(X_3d)} correspondences for PnP")

    # Check if we have enough correspondences
    if len(X_3d) < 6:
        print("Warning: Not enough correspondences for reliable PnP")
        # Don't return here unless this is in a function
        # return None, None, None, None

    camera_pose_3 = LinearPnP(X_3d, x_2d, K)
    C_3, R_3 = camera_pose_3
    print(f"Rotation: {R_3}, Centre: {C_3}")

    R_opt_3, C_opt_3, inliers_3= PnPRANSAC(X_3d, x_2d, K)
    print(f" Rotation after pnp ransac for view3: {R_opt_3}, Centre for view3: {C_opt_3}")

    # Non linear PnP
    final_Copt_3, final_Ropt_3 = NonlinearPnP(X_3d, x_2d, K, R_opt_3, C_opt_3)
    print(f"Rotation and centre for camera 3 after Non linear PnP:{final_Ropt_3}, {final_Copt_3}")



if __name__ == '__main__':
    main()
# Create Your Own Starter Code :)
