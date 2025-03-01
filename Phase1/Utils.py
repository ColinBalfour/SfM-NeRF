import numpy as np

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
