import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import time

from Utils import *
from Fundamental import *
from Triangulation import *
from PnP import *
from BundleAdjustment import *

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


def parse_matching_files(folder_path, num_images):
    """
    Scans 'folder_path' for files named 'matching{i}.txt' (i = 1..num_images),
    parses their contents, and returns:
    
    1) matches_dict:
       {
         (i_ref, j): [
             (u_i, v_i, u_j, v_j, R, G, B, feature_id),
             ...
         ],
         ...
       }
    2) all_feature_points: a list where each element is a dictionary that
       maps image_id -> (u, v, R, G, B). Each entry describes one feature
       (track) seen by possibly multiple images.
    3) visibility_mask_per_image: a list (indexed by image_id) of boolean
       lists. For each image i, visibility_mask_per_image[i][f] is True if
       feature f is observed in image i, otherwise False.
    """
    matches_dict = {}
    all_feature_points = []
    max_image_id = 0
    
    # We'll keep a running index 'f_idx' for each new feature
    f_idx = 0

    for i_ref in range(1, num_images + 1):
        filename = f"matching{i_ref}.txt"
        file_path = os.path.join(folder_path, filename)

        # Skip if file doesn't exist
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as f:
            # First line: "nFeatures: <some_number>"
            line = f.readline().strip()
            n_features = int(line.split(':')[1].strip())

            # For each feature in matching{i_ref}.txt
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                
                # print(_, tokens, n_features, i_ref)

                num_matches = int(tokens[0])
                R, G, B = map(int, tokens[1:4])
                u_i, v_i = map(float, tokens[4:6])

                max_image_id = max(max_image_id, i_ref)

                # Create a new feature dictionary for the i_ref row
                feature_dict = {
                    i_ref: (u_i, v_i)
                }

                # For each matched image
                idx = 6
                # print(tokens)
                for _m in range(num_matches-1):
                    j = int(tokens[idx])
                    u_j = float(tokens[idx + 1])
                    v_j = float(tokens[idx + 2])
                    idx += 3

                    max_image_id = max(max_image_id, j)

                    # 1) Store in matches_dict
                    #    Append the feature_id at the end of the tuple
                    if (i_ref, j) not in matches_dict:
                        matches_dict[(i_ref, j)] = []
                    matches_dict[(i_ref, j)].append(
                        (u_i, v_i, u_j, v_j, f_idx)
                    )

                    # 2) Append info to feature_dict (no color known for j)
                    feature_dict[j] = (u_j, v_j)

                # Add this feature to all_feature_points
                all_feature_points.append(feature_dict)

                # Increment feature ID
                f_idx += 1

    # Build the visibility mask
    # F = len(all_feature_points)
    # visibility_mask_per_image = [ [False]*F for _ in range(max_image_id + 1) ]
    visibility_mask_per_image = np.zeros((max_image_id + 1, f_idx))

    # For each feature f_idx, mark which images see it
    for f_idx, feature_dict in enumerate(all_feature_points):
        for img_id in feature_dict.keys():
            visibility_mask_per_image[img_id][f_idx] = 1

    return matches_dict, all_feature_points, visibility_mask_per_image

def get_keypoints_and_matches_for_pair(matches):
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
        pt1 = feature[:2]
        pt2 = feature[2:4]
        # Create KeyPoint objects (using an arbitrary size, e.g., 5)
        keypoint1 = cv2.KeyPoint(x=pt1[0], y=pt1[1], size=5)
        keypoint2 = cv2.KeyPoint(x=pt2[0], y=pt2[1], size=5)
        kp1.append(keypoint1)
        kp2.append(keypoint2)
        # Create a DMatch object. The indices correspond to the order in the lists.
        match = cv2.DMatch(_queryIdx=len(kp1) - 1, _trainIdx=len(kp2) - 1, _distance=0)
        dmatches.append(match)

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


def refine_all_matches(matches_dict, 
                       all_feature_points,
                       visibility_mask_per_image
                       ):
    """
    1) For each (i, j) in matches_dict, calls reject_outliers_func on the list of matches.
    2) Updates matches_dict to keep only the inliers.
    3) Any outlier match for feature f_idx is removed from all_feature_points[f_idx]
       and from the visibility mask.
    4) Returns the refined matches_dict, all_feature_points, and visibility_mask_per_image.
    """
    refined_matches_dict = {}

    # We'll iterate over each pair in matches_dict
    for (i, j), match_list in matches_dict.items():
        
        print(f"Refining matches for pair ({i}, {j})...")
        
        # 1) Apply the outlier rejection
        # Apply outlier rejection
        arr = np.array(match_list)
        kp1, kp2, dmatches= get_keypoints_and_matches_for_pair(match_list)
        
        F, inliers = reject_outliers(kp1, kp2, dmatches)
        
        refined_list = arr[inliers] 
        
        # 2) Store the refined list in a new dictionary
        refined_matches_dict[(i, j)] = F, refined_list
        
        # 3) Find the matches that were deemed outliers
        mask = ~np.isin(arr, refined_list).all(axis=1)
        outliers = arr[mask]
        
        # 4) For each outlier, remove references from all_feature_points and the mask
        for outlier in outliers:
            # outlier is (u_i, v_i, u_j, v_j, f_idx)
            f_idx = int(outlier[-1])

            # Remove i's observation from the feature dictionary
            if i in all_feature_points[f_idx]:
                del all_feature_points[f_idx][i]
                visibility_mask_per_image[i][f_idx] = 0
            
            # Remove j's observation from the feature dictionary
            if j in all_feature_points[f_idx]:
                del all_feature_points[f_idx][j]
                visibility_mask_per_image[j][f_idx] = 0

            # If the feature dictionary is now empty, that means no images see it
            # We could leave it as an empty dict, or set it to None, etc.
            if len(all_feature_points[f_idx]) == 0:
                all_feature_points[f_idx] = {}  # or None, if you prefer

    return refined_matches_dict, all_feature_points, visibility_mask_per_image


def visualize_3d_points(X_final, C_final, X_optimized=None,  title="3D Points Visualization"):
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

    # For the second camera (if you have the position)
    # Use C_final which is the camera center
    if 'C_final' in globals():
        ax.scatter(C_final[0], C_final[1], C_final[2], c='purple', marker='s', s=100, label='Camera 2')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add a legend
    ax.legend()

    # Set reasonable limits based on data
    # You might need to adjust these based on your specific data
    max_range = np.max([np.max(np.abs(X)), np.max(np.abs(Y)), np.max(np.abs(Z))])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Display the plot
    plt.tight_layout()
    plt.savefig('3d_points_visualization.png', dpi=300)
    plt.show()
    
    
def visualize_reconstruction(X_all, camera_info):
    """
    Visualize the complete 3D reconstruction with all cameras.

    Parameters:
    X_all: All 3D points (N x 3)
    camera_info: Dictionary of camera positions and rotations
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(X_all[:, 0], X_all[:, 1], X_all[:, 2],
               c='blue', marker='.', s=2, alpha=0.6)

    # Plot camera positions
    for i, info in camera_info.items():
        C = info['C']
        R = info['R']
        ax.scatter(C[0], C[1], C[2], color=f'C{i}', marker='s', s=100,
                   label=f'Camera {i}')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Complete 3D Reconstruction')
    ax.legend()

    # Adjust view limits
    if len(X_all) > 0:
        max_range = np.max([
            np.max(np.abs(X_all[:, 0])),
            np.max(np.abs(X_all[:, 1])),
            np.max(np.abs(X_all[:, 2]))
        ]) * 1.2  # Add 20% margin

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig('complete_reconstruction.png', dpi=300)
    plt.show()
    

def triangulate(K, F, R1, T1, im1, im2, kp1, kp2, dmatches, show=False, useCV=False):
    """
    Given two images and their matches, estimate the camera pose and triangulated points.

    Parameters:
    im1: First image
    im2: Second image
    K: Camera intrinsic matrix
    matches: List of matches
    
    Returns:
    """

    # Estimate the fundamental matrix
    # F, inliers = reject_outliers(kp1, kp2, dmatches)
    
    fpts1 = np.array([kp1[m.queryIdx].pt for m in dmatches])
    fpts2 = np.array([kp2[m.trainIdx].pt for m in dmatches])
    # F = estimate_fundamental_matrix(fpts1, fpts2)
    
    # print("ahhh")
    # print(F)
    # raise Exception

    if show:
        print("Estimated fundamental matrix F:")
        print(F)

    E = get_essential_mtx(K, F)
    if show:
        print("Essential matrix E:")
        print(E)

    pose = get_camera_pose(E)
    if show:
        print(f"Camera pose {pose}")
        

    # For each possible pose
    triangulated_points = []

    for C, R in pose:
        # Convert camera center to translation
        T2 = -np.dot(R, C.reshape(3, 1))

        # print(R1, T1)
        # Triangulate points
        points_3d = triangulationlinear(K, R1, T1, R, T2, fpts1, fpts2)
        triangulated_points.append(points_3d)

    X_final, C_final, R_final = chirality_condition(triangulated_points, pose)
    
    # if useCV:
    #     _, R_cv, t_cv ,_ = cv2.recoverPose(E, fpts1, fpts2)
    #     R_final = R_cv
    #     C_final = t_cv.reshape(3,1)
        
    #     idx = np.argmin([np.linalg.norm(t_cv - C.reshape(3,1)) + np.linalg.norm(R_cv - R) for C, R in pose])
    #     X_final = np.array(triangulated_points[idx])
    
    # print("triangulated points:", triangulated_points)
    print("number of triangulated points:", len(X_final))
    
    colors = ['blue', 'green', 'red', 'orange']
    plt.figure(figsize=(10, 8))
    # For each camera pose solution
    for i, points in enumerate(triangulated_points):
        if len(points) == 0:
            continue

        # Convert to numpy array if it's not already
        points_array = np.array(points)

        # Extract x and z coordinates
        x_coords = points_array[:, 0]
        z_coords = points_array[:, 2]

        # Plot with a specific color
        color = colors[i % len(colors)]
        plt.scatter(x_coords, z_coords, color=color, s=10, alpha=0.7,
                    label=f'Camera pose {i + 1}')

    # Configure the plot to match your example
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('X vs Z Coordinates for Different Camera Poses')
    plt.legend()

    # Add equal aspect ratio to maintain proper scaling
    plt.axis('equal')

    # Save and show
    plt.savefig('x_vs_z_triangulation.png', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
        
    if show:
        _, R_cv, t_cv ,_ = cv2.recoverPose(E, fpts1, fpts2)
        print("R_cv:", R_cv)
        print("t_cv:", t_cv)
        print("best_R", R_final)
        print("best_T", C_final)

        #Reprojection error after linear triangulation
        error_1, error_2, reproj_errors = mean_reprojection_error(fpts1, fpts2, X_final, R1, T1, R_final, C_final, K)
        print(f"Mean Reprojection error after linear triangulation error: {reproj_errors}")
        print(f"Reprojection error after linear triangulation frame 1: {error_1}")
        print(f"Reprojection error after linear triangulation frame 2: {error_2}")
        
        # Projected Points after linear triangulation
        projected_pts_frame1, projected_pts_frame2  = projectedpointframe(R1, T1, R_final, C_final, K, X_final)
        # Draw projected points on frame 1
        img1_with_points = im1.copy()
        for pt in projected_pts_frame1:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img1_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles
        # Draw original matched points on frame 1
        for pt in fpts1:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img1_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

        # Similarly for frame 2
        img2_with_points = im2.copy()  # Assuming images[1] is frame 2
        for pt in projected_pts_frame2:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img2_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles

        for pt in fpts2:
            x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

        # Display the images
        cv2.imshow("Frame 1 after linear triangulation- Green: Projected, Red: Original", img1_with_points)
        cv2.imshow("Frame 2 after linear triangulation  - Green: Projected, Red: Original", img2_with_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        cv2.imwrite("Frame1 - lineartriangulation.jpg", img1_with_points)
        cv2.imwrite("Frame2 - lineartriangulation.jpg", img2_with_points)

    # Nonlinear triangulation
    X_optimized = non_linear_triangulation(K, R1, T1, R_final, C_final, fpts1, fpts2, X_final)
    X_optimized = np.array(X_optimized)
    # Check the shape

    if show:
        # Reprojection error after non linear triangulation
        error_frame1, error_frame2, reproj_error  = mean_reprojection_error(fpts1, fpts2, X_optimized, R1, T1, R_final, C_final, K)
        print(f"Mean Reprojection error after non linear triangulation error: {reproj_error}")
        print(f"Reprojection error after non linear triangulation frame 1: {error_frame1}")
        print(f"Reprojection error after non linear triangulation frame 1: {error_frame2}")

        # Projection after non_linear triangulation
        proj_frame1, proj_frame2 = projectedpointframe(R1, T1, R_final, C_final, K, X_optimized)

        # Draw projected points on frame 1
        img1_with_points = im1.copy()
        for pt in proj_frame1:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img1_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles

        # Draw original matched points on frame 1
        for pt in fpts1:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img1_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

        #  frame 2
        img2_with_points = im2.copy()  # Assuming images[1] is frame 2
        for pt in proj_frame2:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img2_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles

        for pt in fpts2:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img2_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

        # Display the images
        cv2.imshow("Frame 1 after non-linear triangulation - Green: Projected, Red: Original", img1_with_points)
        cv2.imshow("Frame 2  after non-linear triangulation- Green: Projected, Red: Original", img2_with_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        cv2.imwrite("Frame1 - nonlineartriangulation.jpg", img1_with_points)
        cv2.imwrite("Frame2 -non lineartriangulation.jpg", img2_with_points)


    # Call this function after triangulation and optimization
    if show:
        visualize_3d_points(X_final, C_final, X_optimized)
        
    # if useCV:
    #     ret, R, t, mask, points = cv2.recoverPose(E, fpts1, fpts2, K, 20, np.ones(len(fpts1)))
    #     return t, R, points
    
    return C_final, R_final, X_optimized


def get_pose(i, obj_points, img_points, K):
    try:
        R_init, C_init, inliers_pnp = PnPRANSAC(obj_points, img_points, K)

        if len(inliers_pnp) < 6:
            print(f"Not enough inliers for reliable PnP with image {i}")
            return None, None

        # Calculate reprojection error after linear PnP
        errorLinearPnP = reprojectionErrorPnP(obj_points[inliers_pnp], img_points[inliers_pnp], K, R_init, C_init)

        # Non-linear refinement of camera pose
        Ci, Ri = NonlinearPnP(obj_points[inliers_pnp], img_points[inliers_pnp], K, R_init, C_init)
        # print(Ri, Ci)
        errorNonLinearPnP = reprojectionErrorPnP(obj_points[inliers_pnp], img_points[inliers_pnp], K, Ri, Ci)
        print(f"Error after linear PnP: {errorLinearPnP}, Error after non-linear PnP: {errorNonLinearPnP}")

        
    except Exception as e:
        print(f"Error in PnP for image {i}: {e}")
        raise e
        return None, None
    
    return Ci, Ri
     


def main():
    # Set data folder and number of images
    # path = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    path = "Phase1/Data"
    num_imgs = 5

    # Load images (which are already undistorted and resized to 800x600px)
    images = load_images(path, num_imgs)
    if len(images) < 2:
        print("Need at least two images to match.")
        return
    # path2 = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    path2 = "Phase1/Data"

    # Load camera calibration parameters (intrinsic matrix K) if needed
    calib_file = os.path.join(path2, "calibration.txt")
    try:
        K = load_calibration(calib_file)
        print("Camera intrinsic matrix K:")
        print(K)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return
    
    # Load matching files
    matches_dict, all_feature_points, visibility_mask = parse_matching_files(path2, len(images))
    refined_matches, all_feature_points, visibility_mask = refine_all_matches(matches_dict, all_feature_points, visibility_mask)
    
    # Get the first two images and their matches
    img1 = images[0]
    img2 = images[1]
    F12, matches = refined_matches[(1, 2)]
    
    kp1, kp2, dmatches = get_keypoints_and_matches_for_pair(matches)
    display_matches(img1, img2, kp1, kp2, dmatches)
    # Triangulate
    camera_info = {
        1: {
            'R': np.eye(3),
            'C': np.zeros((3, 1)),
        }
    }
    C, R, X12 = triangulate(K, F12, camera_info[1]['R'], camera_info[1]['C'], img1, img2, kp1, kp2, dmatches, show=True)
    camera_info[2] = {'R': R, 'C': C}
    
    # print(len(matches))
    # print(len(X12))
    # raise Exception
    
    fIdx_to_3D = {}
    for idx, match in enumerate(matches):
        # match is (u1, v1, u2, v2, f_idx)
        f_idx = int(match[-1])
        fIdx_to_3D[f_idx] = X12[idx]  # one 3D point per match
    
    for i in range(3, len(images)+1):
        print(f'Registering Image: {i} ......')
        
        # Get world and image points for PnP
        obj_points = []
        img_points = []
        
        #NOTE: better (computationally) way to do this might be take fIdx.keys() and take intersection with visibility_mask[i]
        # but eh it works for now so

        # For each feature f_idx that we have from triangulation:
        for f_idx, xyz in fIdx_to_3D.items():
            # Check if image i sees this feature (via all_feature_points)
            if i in all_feature_points[f_idx]:
                # The stored tuple (u, v)
                # Adjust indexing if needed
                u_i, v_i = all_feature_points[f_idx][i][:2]

                obj_points.append(xyz)       # 3D coordinates of the feature
                img_points.append([u_i, v_i])  # 2D pixel coordinates
        
        if len(obj_points) < 8:
            print(f"  Not enough 2D-3D correspondences to solve PnP for image {i}. Skipping.")
            continue
        
        obj_points = np.array(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.array(img_points, dtype=np.float32).reshape(-1, 2)
        
        # NOTE: get_pose will filter out (RANSAC) certain object points and image points, but *we don't use that*
        # This should probably output the filtered points and we should remove the outliers, but we don't do that here 
        C, R = get_pose(i, obj_points, img_points, K)
        if C is None or R is None:
            print(f"  Could not estimate camera pose for image {i}. Skipping.")
            # time.sleep(2)
            return
        # Store camera pose
        camera_info[i] = {'R': R, 'C': C}
        
        

        # Now triangulate new points between this camera and all previous cameras
        for j in range(1, i):  # For all previous cameras
            print(f"  Triangulating between {i} and {j} ...")
            img1 = images[j-1]
            img2 = images[i-1]
            
            pair_key = (j, i)
            if pair_key not in refined_matches:
                print(f"No matches for pair {pair_key}. Skipping.")
                continue
            
            F, matches = refined_matches[pair_key]
            kp1, kp2, dmatches = get_keypoints_and_matches_for_pair(matches)
            
            # Triangulate
            C1 = camera_info[j]['C']
            R1 = camera_info[j]['R']
            if C1 is None or R1 is None:
                print(f"  Could not find camera pose for image {j}. Skipping.")
                continue
            _, _, Xnew = triangulate(K, F, R1, C1, img1, img2, kp1, kp2, dmatches, show=False)
            
            
            # Each row in Xnew corresponds to matches[idx], which is (u_j,v_j, u_i,v_i, f_idx)
            for idx2, match in enumerate(matches):
                f_idx = int(match[-1])
                
                # NOTE: would 3d point be more accurate from this pose? Should it be updated anyways?
                # I feel like prob not since if its already stored it should be closer in index so feature matches are better...
                
                # If we haven't already stored this feature ID's 3D coords, store it now
                if f_idx not in fIdx_to_3D:
                    fIdx_to_3D[f_idx] = Xnew[idx2]
            
            
        print(f'Registered Camera: {i}')
        
    # Visualize the complete 3D reconstruction
    all_world_points = np.array(list(fIdx_to_3D.values()))
    visualize_reconstruction(all_world_points, camera_info)
    
    # Create a 2D top-down view (X-Z plane)
    plt.figure(figsize=(10, 10))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    if len(all_world_points) > 0:
        plt.scatter(all_world_points[:, 0], all_world_points[:, 2], marker='.', linewidths=0.5, color='blue')

    # Plot camera positions
    for i, info in camera_info.items():
        C = info['C']
        R = info['R']
        plt.plot(C[0], C[2], marker='o', markersize=15, linestyle='None',
                 label=f'Camera {i}')

    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Top-down View (X-Z Plane)')
    plt.legend()
    plt.savefig('topdown_view.png')
    plt.show()
    
    plt.close()

    print("Bundle adjustment...")
    
    
        
    # Suppose you have N cameras, M points
    N = len(camera_info)    # e.g. cameras labeled 1..N, gather them in sorted order
    M = len(all_feature_points)  # total features

    # 1) Build camera_params_init (N x 6) and points_3d_init (M x 3)
    camera_params_init = np.zeros((N, 6), dtype=np.float32)
    points_3d_init = np.zeros((M, 3), dtype=np.float32)
    points_2d = np.full((N, M, 2), -1, dtype=np.float32)  # or something similar
    print(points_2d.shape)

    # 'camera_ids' is a sorted list of the cameras in your pipeline
    camera_ids = sorted(list(camera_info.keys()))  # e.g. [1,2,3,...]

    # Fill camera_params_init
    for i, cam_id in enumerate(camera_ids):
        R = camera_info[cam_id]['R']
        C = camera_info[cam_id]['C']  # camera center in world coords
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R @ C  # world->camera
        camera_params_init[i, :3] = rvec.ravel()
        camera_params_init[i, 3:] = tvec.ravel()

    # Fill points_3d_init and points_2d, plus build a visibility matrix
    # e.g. if fIdx_to_3D[f] = [X, Y, Z], fill points_3d_init[f, :] = ...
    # And if feature f is in camera camera_ids[i], fill points_2d[i, f] = (u,v)

    visibility_matrix = np.zeros((N, M), dtype=np.uint8)

    for f_idx, feature_dict in enumerate(all_feature_points):
        # Suppose we have points_3d_init[f_idx] from fIdx_to_3D
        if f_idx in fIdx_to_3D:
            points_3d_init[f_idx] = fIdx_to_3D[f_idx]
        
        for cam_id, observation in feature_dict.items():
            if cam_id in camera_ids:
                i = camera_ids.index(cam_id)
                u, v = observation[0], observation[1]
                points_2d[i, f_idx] = (u, v)
                visibility_matrix[i, f_idx] = 1

    # 2) Call bundle adjustment
    cam_params_opt, points_3d_opt, cost, success = bundle_adjustment(
        K,
        camera_params_init,
        points_3d_init,
        points_2d,
        visibility_matrix
    )

    if success:
        print(f"Bundle adjustment converged. Final cost: {cost}")
    else:
        print("Bundle adjustment did not converge.")

    # 3) Convert 'cam_params_opt' back to R, C
    for i, cam_id in enumerate(camera_ids):
        rvec_opt = cam_params_opt[i, :3]
        tvec_opt = cam_params_opt[i, 3:]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        C_opt = -R_opt.T @ tvec_opt
        camera_info[cam_id]['R'] = R_opt
        camera_info[cam_id]['C'] = C_opt

    # 4) Update your 3D points
    for f_idx in range(M):
        X, Y, Z = points_3d_opt[f_idx]
        fIdx_to_3D[f_idx] = np.array([X, Y, Z])
        
    # Visualize the complete 3D reconstruction
    all_world_points = np.array(list(fIdx_to_3D.values()))
    visualize_reconstruction(all_world_points, camera_info)
    
    # Create a 2D top-down view (X-Z plane)
    plt.figure(figsize=(10, 10))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    if len(all_world_points) > 0:
        plt.scatter(all_world_points[:, 0], all_world_points[:, 2], marker='.', linewidths=0.5, color='blue')

    # Plot camera positions
    for i, info in camera_info.items():
        C = info['C']
        R = info['R']
        plt.plot(C[0], C[2], marker='o', markersize=15, linestyle='None',
                 label=f'Camera {i}')

    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Top-down View (X-Z Plane)')
    plt.legend()
    plt.savefig('topdown_view.png')
    plt.show()

    print("Done")
                    
        

if __name__ == '__main__':
    main()

