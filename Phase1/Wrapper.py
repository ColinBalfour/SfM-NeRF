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
    #P = np.dot(K, np.dot(R, np.hstack((I, -T))))
    P = np.dot(K, np.hstack((R, T)))


    # Project point
    x_proj_homo = np.dot(P, X_homo)

    # Convert from homogeneous to image coordinates
    x_proj = x_proj_homo[:2] / x_proj_homo[2]

    return x_proj[0], x_proj[1]

def projectedpointframe(R, T, R_final, C_final, K, X_final):
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
       # x, y = project_point_to_image(point, R_final, C_final, K)
        x, y = project_point_to_image(point, R_final, T2, K)
        projected_pts_frame2.append((x, y))

    return np.array(projected_pts_frame1), np.array(projected_pts_frame2)


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



def main():
    # Set data folder and number of images
    path = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    #path = "Data"
    num_imgs = 5

    # Load images (which are already undistorted and resized to 800x600px)
    images = load_images(path, num_imgs)
    if len(images) < 2:
        print("Need at least two images to match.")
        return
    path2 = "D:/Computer vision/Homeworks/5. Project 2 - Phase1/YourDirectoryID_p2 (1)/YourDirectoryID_p2/Phase1/P2Data/P2Data"
    #path2 = "Data"

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
    target_img_id = 2 # Change this if you want to display matches with a different image.
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

    # For each possible pose
    triangulated_points = []

    # First camera is at origin
    R1 = np.identity(3)
    T1 = np.zeros((3, 1))

    Camera_orientation = []
    Camera_orientation.append((T1,R1))

    for C, R in pose:
        # Convert camera center to translation
        T2 = -np.dot(R, C.reshape(3, 1))

        # Triangulate points
        points_3d = triangulationlinear(K, R1, T1, R, T2, fpts1, fpts2)
        triangulated_points.append(points_3d)

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
    print("triangulated points:", triangulated_points)
    print("length of triangulated points:", len(triangulated_points))

   #  # Linear triangulation
   #  # Rotation and translation of base frame
    R = np.identity(3)
    T = np.zeros((3, 1))
   #  #X_final, R_final, C_final, wp = linear_triangulation(K, R, T, pose, fpts1, fpts2)
   #  X_final, camera_pose, wp= linear_triangulation(K, R, T, pose, fpts1, fpts2)
   #  R_final = np.array(camera_pose[1]).reshape(3,3)
   #  C_final = np.array(camera_pose[0]).reshape(3, 1)
   # # C_final = - C_final
   # #  translation = - np.dot(R_final, C_final)
   # #  print(f"Translation {translation}")
   #  # Define colors for different solutions
    colors = ['blue', 'green', 'red', 'orange']
    # Place this code in your plotting section
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

    # Configure the plot
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('X vs Z Coordinates for Different Camera Poses')
    plt.legend()

    # Set the x-axis limits from -10 to 10
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # If you also want to set the y-axis (z-coordinate) limits
    # plt.ylim(-10, 10)

    # Save and show
    plt.savefig('x_vs_z_triangulation.png', dpi=300)
    plt.show()
    # plt.figure(figsize=(10, 8))
    # # For each camera pose solution
    # for i, points in enumerate(triangulated_points):
    #     if len(points) == 0:
    #         continue
    #
    #     # Convert to numpy array if it's not already
    #     points_array = np.array(points)
    #
    #     # Extract x and z coordinates
    #     x_coords = points_array[:, 0]
    #     z_coords = points_array[:, 2]
    #
    #     # Plot with a specific color
    #     color = colors[i % len(colors)]
    #     plt.scatter(x_coords, z_coords, color=color, s=10, alpha=0.7,
    #                 label=f'Camera pose {i + 1}')
    #
    # # Configure the plot to match your example
    # plt.grid(True)
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.title('X vs Z Coordinates for Different Camera Poses')
    # plt.legend()
    #
    # # Add equal aspect ratio to maintain proper scaling
    # plt.axis('equal')
    #
    # # Save and show
    # plt.savefig('x_vs_z_triangulation.png', dpi=300)
    # plt.show()

    _, R_cv, t_cv ,_ = cv2.recoverPose(E, fpts1, fpts2)
    print("R_cv:", R_cv)
    print("t_cv:", t_cv)
    print("best_R", R_final)
    print("best_T", C_final)

    #Reprojection error after linear triangulation
    error_1, error_2, reproj_errors = mean_reprojection_error(fpts1, fpts2, X_final, R, T, R_final, C_final, K)
    print(f"Mean Reprojection error after linear triangulation error: {reproj_errors/len(fpts1)}")
    print(f"Reprojection error after linear triangulation frame 1: {error_1/len(fpts1)}")
    print(f"Reprojection error after linear triangulation frame 2: {error_2/len(fpts2)}")


    # Projected Points after linear triangulation
    projected_pts_frame1, projected_pts_frame2  = projectedpointframe(R, T, R_final, C_final, K, X_final)
    # Draw projected points on frame 1
    img1_with_points = images[0].copy()
    for pt in projected_pts_frame1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles
    # Draw original matched points on frame 1
    for pt in fpts1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    # Similarly for frame 2
    img2_with_points = images[1].copy()  # Assuming images[1] is frame 2
    for pt in projected_pts_frame2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles

    for pt in fpts2:
        x, y = int(round(pt[0])), int(round(pt[1]))
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
    # Check the shape

    # Reprojection error after non linear triangulation
    error_frame1, error_frame2, reproj_error  = mean_reprojection_error(fpts1, fpts2, X_optimized, R, T, R_final, C_final, K)
    print(f"Mean Reprojection error after non linear triangulation error: {reproj_error/len(fpts1)}")
    print(f"Reprojection error after non linear triangulation frame 1: {error_frame1/len(fpts1)}")
    print(f"Reprojection error after non linear triangulation frame 1: {error_frame2/len(fpts2)}")



    # Projection after non_linear triangulation
    proj_frame1, proj_frame2 = projectedpointframe(R, T, R_final, C_final, K, X_optimized)

    # Draw projected points on frame 1
    img1_with_points = images[0].copy()
    for pt in proj_frame1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles

    # Draw original matched points on frame 1
    for pt in fpts1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img1_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    #  frame 2
    img2_with_points = images[1].copy()  # Assuming images[1] is frame 2
    for pt in proj_frame2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 2, (0, 255, 0), -1)  # Green circles

    for pt in fpts2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img2_with_points, (x, y), 2, (0, 0, 255), -1)  # Red circles

    # Display the images
    cv2.imshow("Frame 1 after non-linear triangulation - Green: Projected, Red: Original", img1_with_points)
    cv2.imwrite("Frame1 - nonlineartriangulation.jpg", img1_with_points)
    cv2.imshow("Frame 2  after non-linear triangulation- Green: Projected, Red: Original", img2_with_points)
    cv2.imwrite("Frame2 -non lineartriangulation.jpg", img2_with_points)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    Camera_orientation.append((C_final, R_final))

    # Call this function after triangulation and optimization
    visualize_3d_points(X_final, C_final, X_optimized)
    # Create scatter plot comparing original and optimized Z values


    # Camera 3rd image
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
    threshold = 1.0

    # Keep track of indices for visualization
    matched_indices = []

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
                matched_indices.append((i, j))
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
    print(f" Rotation after non linear pnp ransac for view3: {final_Ropt_3}, Centre for view3: {final_Copt_3}")

    # Updating world point using linear triangulation
    points3d = triangulationlinear(K, R, T, final_Ropt_3, final_Copt_3, pts1_img3, pts3)
    X = np.array(points3d)



if __name__ == '__main__':
    main()
