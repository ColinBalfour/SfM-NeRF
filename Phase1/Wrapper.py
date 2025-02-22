import cv2
import numpy as np
import os

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
    for i in range(1, num_imgs+1):
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
            u = float(tokens[offset+1])
            v = float(tokens[offset+2])
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
                match = cv2.DMatch(_queryIdx=len(kp1)-1, _trainIdx=len(kp2)-1, _distance=0)
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
        A[i, :] = np.array([ x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1 ])
        
    # Solve for F using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set smallest singular value to zero
    F = U @ np.diag(S) @ Vt  # Reconstruct F with rank-2 constraint
    
    F = F / F[2,2] # enforce F(3,3) = 1
    
    # print("custom:")
    # print(F)
    # print(np.linalg.matrix_rank(F))
    
    
    return F

def reject_outliers(kp1, kp2, dmatches, N=5000, threshold=.01):
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
    return best_F, best_inliers
    
def get_essential_mtx(K, F):
    """
    Compute the essential matrix from the fundamental matrix and camera intrinsics.
    """
    
    E = K.T @ F @ K
    return E
    
    

def main():
    # Set data folder and number of images
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
    
    E = get_essential_mtx(K, F)
    print("Estimated essential matrix E:")
    print(E)

if __name__ == '__main__':
    main()
