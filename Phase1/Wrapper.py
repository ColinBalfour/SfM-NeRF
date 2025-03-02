import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from Utils import *
from Fundamental import *
from Triangulation import *
from PnP import *


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


def get_keypoints_and_matches_for_pair(matches, source_img_id, target_img_id):
    """
    Given a list of features, extract keypoints and matching points between any two images.
    Also returns the feature indices for each match.

    Parameters:
    matches: List of matches from the matching file
    source_img_id: Source image ID (1-based)
    target_img_id: Target image ID (1-based)

    Returns:
    kp1: list of cv2.KeyPoint for the source image
    kp2: list of cv2.KeyPoint for the target image
    dmatches: list of cv2.DMatch linking kp1 to kp2
    feature_indices: list of original feature indices
    """
    kp1 = []
    kp2 = []
    dmatches = []
    feature_indices = []  # Track original feature indices

    for feature_idx, feature in enumerate(matches):
        # Find if this feature appears in both source_img_id and target_img_id
        source_pt = None
        target_pt = None

        # Check if feature appears in source image
        if source_img_id == 1:
            source_pt = feature['pt_curr']
        else:
            for m in feature['matches']:
                if m['image_id'] == source_img_id:
                    source_pt = m['pt']
                    break

        # Check if feature appears in target image
        for m in feature['matches']:
            if m['image_id'] == target_img_id:
                target_pt = m['pt']
                break

        # If feature appears in both images, add to keypoints and matches
        if source_pt is not None and target_pt is not None:
            keypoint1 = cv2.KeyPoint(x=source_pt[0], y=source_pt[1], size=5)
            keypoint2 = cv2.KeyPoint(x=target_pt[0], y=target_pt[1], size=5)
            kp1.append(keypoint1)
            kp2.append(keypoint2)
            match = cv2.DMatch(_queryIdx=len(kp1) - 1, _trainIdx=len(kp2) - 1, _distance=0)
            dmatches.append(match)
            feature_indices.append(feature_idx)  # Store the feature index

    return kp1, kp2, dmatches, feature_indices


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


def get_feature_position_in_image(feature_idx, matches, image_id):
    """
    Get the 2D position of a feature in a specific image.

    Parameters:
    feature_idx: Index of the feature in the matches list
    matches: List of matches from the parsing function
    image_id: Image ID (1-based)

    Returns:
    (x, y) coordinates or None if the feature is not visible in that image
    """
    if feature_idx >= len(matches):
        return None

    feature = matches[feature_idx]

    if image_id == 1:
        return feature['pt_curr']

    # For other images, search in the matches list
    for m in feature['matches']:
        if m['image_id'] == image_id:
            return m['pt']

    return None


def visualize_reconstruction(X_all, X_found, C_set, R_set):
    """
    Visualize the complete 3D reconstruction with all cameras.

    Parameters:
    X_all: All 3D points (N x 3)
    X_found: Binary flags indicating which points have been triangulated (N x 1)
    C_set: List of camera centers
    R_set: List of camera rotation matrices
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get valid 3D points
    valid_indices = np.where(X_found[:, 0] == 1)[0]
    valid_points = X_all[valid_indices]

    # Plot points
    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
               c='blue', marker='.', s=2, alpha=0.6)

    # Plot camera positions
    for i, (C, R) in enumerate(zip(C_set, R_set)):
        ax.scatter(C[0], C[1], C[2], color=f'C{i}', marker='s', s=100,
                   label=f'Camera {i + 1}')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Complete 3D Reconstruction')
    ax.legend()

    # Adjust view limits
    if len(valid_points) > 0:
        max_range = np.max([
            np.max(np.abs(valid_points[:, 0])),
            np.max(np.abs(valid_points[:, 1])),
            np.max(np.abs(valid_points[:, 2]))
        ]) * 1.2  # Add 20% margin

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig('complete_reconstruction.png', dpi=300)
    plt.show()


def visualize_3d_points(X_final, C_final, X_optimized=None, title="3D Points Visualization"):
    """
    Create a 3D scatter plot to visualize triangulated points.

    Parameters:
    X_final: Original triangulated points from linear triangulation
    X_optimized: Optimized points from non-linear optimization (optional)
    title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z coordinates
    X = X_final[:, 0]
    Y = X_final[:, 1]
    Z = X_final[:, 2]

    # Plot original points
    ax.scatter(X, Y, Z, c='blue', marker='o', label='Linear Triangulation', alpha=0.6)

    # Plot optimized points if provided
    if X_optimized is not None:
        X_opt = X_optimized[:, 0]
        Y_opt = X_optimized[:, 1]
        Z_opt = X_optimized[:, 2]
        ax.scatter(X_opt, Y_opt, Z_opt, c='red', marker='^', label='Non-Linear Optimization', alpha=0.6)

    # Add camera positions
    # For the first camera at the origin
    ax.scatter(0, 0, 0, c='green', marker='s', s=100, label='Camera 1')

    # For the second camera
    if isinstance(C_final, np.ndarray):
        ax.scatter(C_final[0], C_final[1], C_final[2], c='purple', marker='s', s=100, label='Camera 2')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add a legend
    ax.legend()

    # Set reasonable limits based on data
    max_range = np.max([np.max(np.abs(X)), np.max(np.abs(Y)), np.max(np.abs(Z))])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Display the plot
    plt.tight_layout()
    plt.savefig('3d_points_visualization.png', dpi=300)
    plt.show()


def main():
    # Set data folder and number of images
    path = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    # path = "Data"
    num_imgs = 5

    # Load images (which are already undistorted and resized to 800x600px)
    images = load_images(path, num_imgs)
    if len(images) < 2:
        print("Need at least two images to match.")
        return
    path2 = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    # path2 = "Data"

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
        n_features, matches = parse_matching_file(matching_file)
    except Exception as e:
        print(f"Error parsing matching file: {e}")
        return
    
    # Create structures to store 3D points and track which ones have been reconstructed
    X_all = np.zeros((n_features, 3))  # 3D point coordinates
    X_found = np.zeros((n_features, 1), dtype=int)  # Binary flag if point has been triangulated
    camera_indices = np.zeros((n_features, 1), dtype=int)  # Which camera observed this point first

    # Get matches between image 1 and image 2
    kp1, kp2, dmatches, feature_indices_12 = get_keypoints_and_matches_for_pair(matches, 1, 2)
    print(f"Found {len(kp1)} matches between image 1 and image 2")

    # Display the matches
    display_matches(images[0], images[1], kp1, kp2, dmatches)

    # Estimate the fundamental matrix
    F, inliers = reject_outliers(kp1, kp2, dmatches)
    print("Estimated fundamental matrix F:")
    print(F)

    # Display the inlier matches
    inlier_matches = [dmatches[i] for i in inliers]
    # Map inliers to feature indices
    inlier_feature_indices = [feature_indices_12[i] for i in inliers]

    display_matches(images[0], images[1], kp1, kp2, inlier_matches)

    # Get point correspondences
    fpts1 = np.array([kp1[m.queryIdx].pt for m in inlier_matches])
    fpts2 = np.array([kp2[m.trainIdx].pt for m in inlier_matches])

    # Compute essential matrix and camera poses
    E = get_essential_mtx(K, F)
    print("Estimated essential matrix E:")
    print(E)

    pose = get_camera_pose(E)
    print(f"Camera pose {pose}")

    # First camera is at origin
    R1 = np.identity(3)
    T1 = np.zeros((3, 1))

    # Triangulate points for each possible camera pose
    triangulated_points = []
    for C, R in pose:
        # Convert camera center to translation
        T2 = -np.dot(R, C.reshape(3, 1))
        # Triangulate points
        points_3d = triangulationlinear(K, R1, T1, R, T2, fpts1, fpts2)
        triangulated_points.append(points_3d)

    # Check which pose gives the most points in front of both cameras
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
            print(f"Best pose: {best_pose_idx + 1}")

    # Get the best pose and points
    X_final = np.array(triangulated_points[best_pose_idx])
    C_final = pose[best_pose_idx][0].reshape(3, 1)
    R_final = pose[best_pose_idx][1]

    # Store camera poses
    C_set = []
    R_set = []

    # First camera
    R = np.identity(3)
    T = np.zeros((3, 1))
    C0 = np.zeros((3, 1))
    C_set.append(C0)
    R_set.append(R)

    # Second camera
    C_set.append(C_final)
    R_set.append(R_final)

    print('Registered Cameras 1 and 2')

    # Plot camera poses and triangulated points
    colors = ['blue', 'green', 'red', 'orange']
    plt.figure(figsize=(10, 8))
    for i, points in enumerate(triangulated_points):
        if len(points) == 0:
            continue
        points_array = np.array(points)
        x_coords = points_array[:, 0]
        z_coords = points_array[:, 2]
        color = colors[i % len(colors)]
        plt.scatter(x_coords, z_coords, color=color, s=10, alpha=0.7,
                    label=f'Camera pose {i + 1}')

    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('X vs Z Coordinates for Different Camera Poses')
    plt.legend()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.savefig('x_vs_z_triangulation.png', dpi=300)
    plt.show()

    # Compare with cv2.recoverPose
    _, R_cv, t_cv, _ = cv2.recoverPose(E, fpts1, fpts2)
    print("R_cv:", R_cv)
    print("t_cv:", t_cv)
    print("best_R", R_final)
    print("best_T", C_final)

    # Reprojection error after linear triangulation
    error_1, error_2, reproj_errors = mean_reprojection_error(fpts1, fpts2, X_final, R, T, R_final, C_final, K)
    print(f"Mean Reprojection error after linear triangulation error: {reproj_errors / len(fpts1)}")
    print(f"Reprojection error after linear triangulation frame 1: {error_1 / len(fpts1)}")
    print(f"Reprojection error after linear triangulation frame 2: {error_2 / len(fpts2)}")

    # Visualize projected points after linear triangulation
    projected_pts_frame1, projected_pts_frame2 = projectedpointframe(R, T, R_final, C_final, K, X_final)

    # Draw projected points on frame 1
    img1_with_points = images[0].copy()
    for pt in projected_pts_frame1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img1_with_points.shape[1] and 0 <= y < img1_with_points.shape[0]:
            cv2.circle(img1_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles
    # Draw original matched points on frame 1
    for pt in fpts1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img1_with_points.shape[1] and 0 <= y < img1_with_points.shape[0]:
            cv2.circle(img1_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    # Similarly for frame 2
    img2_with_points = images[1].copy()
    for pt in projected_pts_frame2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img2_with_points.shape[1] and 0 <= y < img2_with_points.shape[0]:
            cv2.circle(img2_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles
    for pt in fpts2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img2_with_points.shape[1] and 0 <= y < img2_with_points.shape[0]:
            cv2.circle(img2_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    # Display the images
    cv2.imshow("Frame 1 after linear triangulation- Green: Projected, Red: Original", img1_with_points)
    cv2.imwrite("Frame1 - lineartriangulation.jpg", img1_with_points)
    cv2.imshow("Frame 2 after linear triangulation  - Green: Projected, Red: Original", img2_with_points)
    cv2.imwrite("Frame2 - lineartriangulation.jpg", img2_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Nonlinear triangulation
    X_optimized = non_linear_triangulation(K, R, T, R_final, C_final, fpts1, fpts2, X_final)
    X_optimized = np.array(X_optimized)

    # Record the triangulated points from the first pair
    for idx, feature_idx in enumerate(inlier_feature_indices):
        if idx < len(X_optimized):
            X_all[feature_idx] = X_optimized[idx, :3]  # Store the optimized 3D point
            X_found[feature_idx] = 1  # Mark as found
            camera_indices[feature_idx] = 1  # First seen in camera 1

    # Reprojection error after non-linear triangulation
    error_frame1, error_frame2, reproj_error = mean_reprojection_error(fpts1, fpts2, X_optimized, R, T, R_final,
                                                                       C_final, K)
    print(f"Mean Reprojection error after non linear triangulation error: {reproj_error / len(fpts1)}")
    print(f"Reprojection error after non linear triangulation frame 1: {error_frame1 / len(fpts1)}")
    print(f"Reprojection error after non linear triangulation frame 2: {error_frame2 / len(fpts2)}")

    # Projection after non-linear triangulation
    proj_frame1, proj_frame2 = projectedpointframe(R, T, R_final, C_final, K, X_optimized)

    # Draw projected points on frame 1
    img1_with_points = images[0].copy()
    for pt in proj_frame1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img1_with_points.shape[1] and 0 <= y < img1_with_points.shape[0]:
            cv2.circle(img1_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles
    for pt in fpts1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img1_with_points.shape[1] and 0 <= y < img1_with_points.shape[0]:
            cv2.circle(img1_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    # Frame 2
    img2_with_points = images[1].copy()
    for pt in proj_frame2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img2_with_points.shape[1] and 0 <= y < img2_with_points.shape[0]:
            cv2.circle(img2_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles
    for pt in fpts2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img2_with_points.shape[1] and 0 <= y < img2_with_points.shape[0]:
            cv2.circle(img2_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    # Display the images
    cv2.imshow("Frame 1 after non-linear triangulation - Green: Projected, Red: Original", img1_with_points)
    cv2.imwrite("Frame1 - nonlineartriangulation.jpg", img1_with_points)
    cv2.imshow("Frame 2 after non-linear triangulation- Green: Projected, Red: Original", img2_with_points)
    cv2.imwrite("Frame2 -non lineartriangulation.jpg", img2_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize 3D points
    visualize_3d_points(X_final, C_final, X_optimized)

    # Register remaining cameras
    for i in range(2, num_imgs):  # Start from the third image (index 2)
        print(f'Registering Image: {i + 1} ......')

        # Find all 3D points that are visible in image i+1
        target_img_id = i + 1
        kp1, kp_i, dmatches_i, feature_indices_i = get_keypoints_and_matches_for_pair(matches, 1, target_img_id)

        # Get inlier matches between image 1 and image i+1
        F_i, inliers_i = reject_outliers(kp1, kp_i, dmatches_i)
        inlier_matches_i = [dmatches_i[idx] for idx in inliers_i]
        inlier_feature_indices_i = [feature_indices_i[idx] for idx in inliers_i]

        # Get point correspondences
        pts1_img_i = np.array([kp1[m.queryIdx].pt for m in inlier_matches_i])
        pts_i = np.array([kp_i[m.trainIdx].pt for m in inlier_matches_i])

        # Find which features have already been triangulated
        X_3d = []
        x_2d = []
        feature_indices_pnp = []

        for idx, feature_idx in enumerate(inlier_feature_indices_i):
            if X_found[feature_idx, 0] == 1:  # If this feature has been triangulated
                X_3d.append(X_all[feature_idx])
                x_2d.append(pts_i[idx])
                feature_indices_pnp.append(feature_idx)

        X_3d = np.array(X_3d)
        x_2d = np.array(x_2d)

        print(f"Found {len(X_3d)} common points between X and image {i + 1}")

        if len(X_3d) < 8:
            print(f"Not enough correspondences for PnP with image {i + 1}")
            continue

        # PnP to estimate camera pose
        try:
            R_init, C_init, inliers_pnp = PnPRANSAC(X_3d, x_2d, K)

            if len(inliers_pnp) < 6:
                print(f"Not enough inliers for reliable PnP with image {i + 1}")
                continue

            # Calculate reprojection error after linear PnP
            errorLinearPnP = reprojectionErrorPnP(X_3d[inliers_pnp], x_2d[inliers_pnp], K, R_init, C_init)

            # Non-linear refinement of camera pose
            Ri, Ci = NonlinearPnP(X_3d[inliers_pnp], x_2d[inliers_pnp], K, R_init, C_init)
            errorNonLinearPnP = reprojectionErrorPnP(X_3d[inliers_pnp], x_2d[inliers_pnp], K, Ri, Ci)
            print(f"Error after linear PnP: {errorLinearPnP}, Error after non-linear PnP: {errorNonLinearPnP}")

            # Store camera pose
            C_set.append(Ci)
            R_set.append(Ri)
        except Exception as e:
            print(f"Error in PnP for image {i + 1}: {e}")
            continue

        # Now triangulate new points between this camera and all previous cameras
        for j in range(i):  # For all previous cameras
            # Get matches between images j+1 and i+1
            kp_j, kp_i, dmatches_ji, feature_indices_ji = get_keypoints_and_matches_for_pair(matches, j + 1, i + 1)

            if len(dmatches_ji) < 8:
                print(f"Not enough matches between images {j + 1} and {i + 1}")
                continue

            # RANSAC for fundamental matrix
            try:
                F_ji, inliers_ji = reject_outliers(kp_j, kp_i, dmatches_ji)
                inlier_matches_ji = [dmatches_ji[idx] for idx in inliers_ji]
                inlier_feature_indices_ji = [feature_indices_ji[idx] for idx in inliers_ji]

                # Get point correspondences
                pts_j = np.array([kp_j[m.queryIdx].pt for m in inlier_matches_ji])
                pts_i = np.array([kp_i[m.trainIdx].pt for m in inlier_matches_ji])

                # Triangulate new points
                T_j = -np.dot(R_set[j], C_set[j].reshape(3, 1))
                T_i = -np.dot(R_set[i], C_set[i].reshape(3, 1))

                X = triangulationlinear(K, R_set[j], T_j, R_set[i], T_i, pts_j, pts_i)
                LT_error = mean_reprojection_error(pts_j, pts_i, X, R_set[j], C_set[j], R_set[i], C_set[i], K)[2] / len(
                    pts_j)

                X_nl = non_linear_triangulation(K, R_set[j], T_j, R_set[i], T_i, pts_j, pts_i, X)
                X_nl = np.array(X_nl)
                nLT_error = mean_reprojection_error(pts_j, pts_i, X_nl, R_set[j], C_set[j], R_set[i], C_set[i], K)[
                                2] / len(pts_j)

                print(
                    f"Error after linear triangulation: {LT_error}, Error after non-linear triangulation: {nLT_error}")

                # Store these points in X_all if they haven't been triangulated yet
                new_points_count = 0
                for idx, feature_idx in enumerate(inlier_feature_indices_ji):
                    if feature_idx < len(X_found) and X_found[feature_idx, 0] == 0:  # If not already triangulated
                        if idx < len(X_nl):
                            # Check if the point is in front of both cameras
                            X_point = X_nl[idx, :3]

                            # Skip points with negative or zero Z
                            if X_point[2] <= 0:
                                continue

                            # Check if point is in front of camera i
                            X_hom = np.append(X_point, 1)
                            X_cam_i = np.dot(R_set[i], X_hom[:3] - C_set[i].flatten())
                            if X_cam_i[2] <= 0:
                                continue

                            X_all[feature_idx] = X_point
                            X_found[feature_idx, 0] = 1
                            new_points_count += 1

                print(f"Triangulated {new_points_count} new points between cameras {j + 1} and {i + 1}")
            except Exception as e:
                print(f"Error in triangulation between images {j + 1} and {i + 1}: {e}")
                continue

        print(f'Registered Camera: {i + 1}')

        # Filter out points with negative Z (behind cameras)
    X_found[X_all[:, 2] <= 0] = 0

    # Count total reconstructed points
    total_points = np.sum(X_found)
    print(f"Total reconstructed points: {total_points}")

    # Visualize the complete reconstruction
    visualize_reconstruction(X_all, X_found, C_set, R_set)

    # Create a 2D top-down view (X-Z plane)
    plt.figure(figsize=(10, 10))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Plot points
    valid_indices = np.where(X_found[:, 0] == 1)[0]
    valid_points = X_all[valid_indices]

    if len(valid_points) > 0:
        plt.scatter(valid_points[:, 0], valid_points[:, 2], marker='.', linewidths=0.5, color='blue')

    # Plot camera positions
    for i, (C, R) in enumerate(zip(C_set, R_set)):
        plt.plot(C[0], C[2], marker='o', markersize=15, linestyle='None',
                 label=f'Camera {i + 1}')

    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Top-down View (X-Z Plane)')
    plt.legend()
    plt.savefig('topdown_view.png')
    plt.show()

    print("Done")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error in main function: {e}")
        traceback.print_exc()
