import numpy as np
import cv2

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

def normalization_matrix(points: np.array) -> np.array:
    """
    Compute the similarity transformation (normalization matrix) that
    translates the centroid of the points to the origin and scales the RMS distance to sqrt(2).
    @ points: (n x 2) array of points.
    @ return: 3x3 normalization matrix.
    """
    centroid = np.mean(points, axis=0)  # Compute centroid (x̄, ȳ)

    # Compute the RMS distance from the centroid
    rms_dist = np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1)))
    scale = np.sqrt(2) / rms_dist
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    return T

def reject_outliers(kp1, kp2, dmatches, N=50000, threshold=.0005, normalize=True):
    """
    Reject outliers using RANSAC.
    """
    # Extract point correspondences
    pts1 = np.array([kp1[m.queryIdx].pt for m in dmatches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in dmatches])
    
    # Get with OpenCV
    cvF, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # print("opencv:")
    # print(cvF)
    # print(np.linalg.matrix_rank(cvF))
    
    if normalize:
    
        normed1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))  # n x 3
        normed2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))  # n x 3
        T1 = normalization_matrix(normed1)  # 3x3
        T2 = normalization_matrix(normed2)  # 3x3
        normed1 = (T1 @ normed1.T).T  # n x 3
        normed2 = (T2 @ normed2.T).T  # n x 3
        
        pts1 = normed1[:, :2]
        pts2 = normed2[:, :2]
        
        # print(normed1[0], normed2[0], normed1[1], normed2[2])

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
        
        h1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        h2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
        
        # Compute the Sampson distance
        Fx1 = F @ h1.T  # 3 x n
        Fx2 = F @ h2.T  # 3 x n
        Fx1 = Fx1 / np.linalg.norm(Fx1[:2], axis=0)
        Fx2 = Fx2 / np.linalg.norm(Fx2[:2], axis=0)
        
        d = np.sum(h2 * Fx1.T, axis=1) ** 2 / (Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2)
        
        # Threshold
        inliers = np.where(d < threshold)[0]
        
        # Count inliers
        # inliers = []
        # for j in range(len(pts1)):
        #     x1 = np.append(pts1[j], 1)
        #     x2 = np.append(pts2[j], 1)
        #     if np.abs(x1.T @ F @ x2) < threshold:
        #         inliers.append(j)

        # Update best F if this iteration has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F

    print(f"RANSAC: Found {len(best_inliers)} inliers out of {len(pts1)} matches.")

    recomputed_F = estimate_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    
    if normalize:
        # Denormalize the fundamental matrix
        recomputed_F = T2.T @ recomputed_F @ T1
    
    return recomputed_F, best_inliers

def get_essential_mtx(K, F):
    """
    Compute the essential matrix from the fundamental matrix and camera intrinsics.
    """

    E = K.T @ F @ K
    return E

def get_camera_pose(E):
    U, D, Vt = np.linalg.svd(E)
    # print("D", D)
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
    R2 = U @ W @ Vt
    R3= U @ W.T @ Vt
    R4 = U @ W.T @ Vt
    pose = [(C1, R1), (C2, R2), (C1, R3), (C2, R4)]
    correct_pose = []
    for C, R in pose:
        if np.linalg.det(R) < 0:
            C = -C
            R = -R
        correct_pose.append((C, R))
    return correct_pose