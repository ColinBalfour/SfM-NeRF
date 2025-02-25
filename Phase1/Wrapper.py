import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import least_squares


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
    C1 = U[:,2]
    C2 = -U[:,2]

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

def get_skew_mat(a):
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

def linear_triangulation(K, R, T, pose, final_pts1, final_pts2):
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R, np.hstack((I, -T))))
    X_all = []  # List to store points for all four solutions
    
    best_pose = 0

    for (C, R2) in pose:
        X_current = []  # Points for current solution
        positive_Z = 0
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
            X_3d = X[:3] / X[3]  # Convert from homogeneous coordinates

            # Store point for current solution
            X_current.append(X_3d)

            # Check cheirality condition
            r3 = R2[:, 2]
            if np.dot(r3, (X_3d - C.flatten())) > 0 and X_3d[2] > 0:
                positive_Z += 1

        # Add all points for current solution to X_all
        X_all.append(X_current)
        if positive_Z > best_pose:
            best_pose = positive_Z
            X_best = X_current
            best_R = R2
            best_T = C

    # Convert lists to numpy arrays
    X_best = np.array(X_best)
    X_all = np.array(X_all)

    return X_best, best_R, best_T, X_all



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

    a = (np.dot(P1_1, X_1)/np.dot(P1_3, X_1))
    b = (np.dot(P1_2, X_1)/np.dot(P1_3, X_1))
    c = (np.dot(P2_1, X_1)/np.dot(P2_3, X_1))
    d = (np.dot(P2_2, X_1)/np.dot(P2_3, X_1))

    geo_error_1 = (a - x1)**2 + (b - y1)**2
    geo_error_2 = (c - x2)**2 + (d - y2)**2
    error = geo_error_1 + geo_error_2

    return error

def non_linear_triangulation(K, R1, T1, R2, T2, finalpts1, finalpts2, X):
    X_optimized = []
    initial_errors = []
    final_errors = []
    for pts1, pts2 , X_i in zip(finalpts1, finalpts2, X):

        #intial error
        initial_residuals = residuals(X_i[:3], K, R1, T1, R2, T2, pts1, pts2)
        initial_error = np.sum(initial_residuals ** 2)
        initial_errors.append(initial_error)

        X_opt = optimization(K, R1, T1, R2, T2, pts1, pts2, X_i)
        X_optimized.append(X_opt)

        #final error
        final_residuals = residuals(X_opt, K, R1, T1, R2, T2, pts1, pts2)
        final_error = np.sum(final_residuals ** 2)
        final_errors.append(final_error)

    # error before optimization
    # initial_error = 0
    # final_error = 0
    # for pts1, pts2, X_1, X_2 in zip(finalpts1, finalpts2, X, X_optimized):
    #     initial_error += np.sum(residuals(X_1, K, R1, T1, R2, T2, pts1, pts2))
    #     final_error += np.sum(residuals(X_2, K, R1, T1, R2, T2, pts1, pts2))
    #
    # print(f"Initial total error: {initial_error}")
    # print(f"Final total error: {final_error}")
        # Print error statistics
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

def plot_dual_view_triangulation(all_world_points, pose=None):
    """
    Create side-by-side plots showing X vs Y (front view) and X vs Z (top view)
    of world points from all four possible triangulation solutions.
    
    R_list, T_list are iterables (lists) of rotation matrices and translation
    vectors for multiple camera poses.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # -- 1. Prepare figure with two subplots side by side --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Colors for different solutions (assuming all_world_points is a list of sets of 3D points)
    colors = ['blue', 'red', 'green', 'cyan']
    
    # -- 2. Plot the triangulated points in both views --
    for i, points in enumerate(all_world_points):
        points = np.array(points)

        # Front view (X vs Y)
        ax1.scatter(points[:, 0], points[:, 1],
                    c=colors[i % len(colors)], s=1, alpha=0.5,
                    label=f'Solution {i + 1}')

        # Top view (X vs Z)
        ax2.scatter(points[:, 0], points[:, 2],
                    c=colors[i % len(colors)], s=1, alpha=0.5)

    
    # -- 3. Loop through each camera pose and plot it --
    if pose is not None:
        R_list = [R for C, R in pose]
        T_list = [C for C, R in pose]
        
        
        #    Prepare a colormap for multiple camera poses --
        #    This creates N distinct colors if we have N cameras.
        num_cameras = len(R_list)
        camera_colors = plt.cm.rainbow(np.linspace(0, 1, num_cameras))
        
        for i, (R, T) in enumerate(zip(R_list, T_list)):
            # Camera center in world coordinates: C = -R^T * T
            print(R.shape, T.shape)
            camera_center = -R.T @ T.reshape(-1)

            # Plot camera center in both views
            ax1.scatter(camera_center[0], camera_center[1],
                        marker='s', c=[camera_colors[i]], s=80,
                        label=f'Camera {i+1}')
            ax2.scatter(camera_center[0], camera_center[2],
                        marker='s', c=[camera_colors[i]], s=80)

            # Draw a small axis or "look" direction to indicate camera orientation.
            # Typically the camera looks along the local Z-axis, i.e. R[:, 2] in world.
            scale = 0.1  # Adjust for visibility
            camera_look_dir = R[:, 2]
            look_endpoint = camera_center + scale * camera_look_dir

            # Front view (X vs Y)
            ax1.plot(
                [camera_center[0], look_endpoint[0]],
                [camera_center[1], look_endpoint[1]],
                color=camera_colors[i],
                linewidth=2
            )

            # Top view (X vs Z)
            ax2.plot(
                [camera_center[0], look_endpoint[0]],
                [camera_center[2], look_endpoint[2]],
                color=camera_colors[i],
                linewidth=2
            )

    # -- 5. Set labels, titles, grid, aspect ratio --
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Front View (X vs Y)')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.axis('equal')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Top View (X vs Z)')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.axis('equal')

    # -- 6. Combine and reduce duplicate legend entries on ax1 --
    handles, labels = ax1.get_legend_handles_labels()
    # Using dict trick to remove duplicates while preserving order
    unique = dict(zip(labels, handles))
    ax1.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    # -- 7. Adjust layout to prevent overlap --
    plt.tight_layout()

    # -- 8. Optionally set some padding around data in both axes --
    for ax in [ax1, ax2]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        x_padding = x_range * 0.1
        y_padding = y_range * 0.1
        ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
        ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

    # -- 9. Show the final figure --
    plt.show()




def main():
    # Set data folder and number of images
    # path = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    path = "Data"
    num_imgs = 5

    # Load images (which are already undistorted and resized to 800x600px)
    images = load_images(path, num_imgs)
    if len(images) < 2:
        print("Need at least two images to match.")
        return

    # Load camera calibration parameters (intrinsic matrix K) if needed
    calib_file = os.path.join(path, "calibration.txt")
    try:
        K = load_calibration(calib_file)
        print("Camera intrinsic matrix K:")
        print(K)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return

    # Load matching data for image1 from matching1.txt
    matching_file = os.path.join(path, "matching1.txt")
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

    #Linear triangulation
    # Rotation and translation of base frame
    R = np.identity(3)
    T = np.zeros((3,1))

    X_final, R_final, T_final, all_world_points = linear_triangulation(K, R, T, pose, fpts1, fpts2)
    print("best_R", R_final)
    print("best_T", T_final)
    print(all_world_points.shape)
    plot_dual_view_triangulation(all_world_points)
    X = all_world_points[3]
    # print("X:", X)
    print("length of X:", len(X))
    X_optimized = non_linear_triangulation(K, R, T, R_final, T_final, fpts1, fpts2, X_final)
    X_optimized = np.array(X_optimized)
    
    plot_dual_view_triangulation(np.array([X_optimized, X_final]))
    



if __name__ == '__main__':
    main()
# Create Your Own Starter Code :)
