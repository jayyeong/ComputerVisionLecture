import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation *3번*
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START

    largest_set = []
    for i in range(10): # repeat ten times
        rand = random.randrange(0, len(matched_pairs)) # sample 랜덤 선택
        choice = matched_pairs[rand]
        orientation = (keypoints1[choice[0]][3] - keypoints2[choice[1]][3]) % (2 * math.pi) # calculation first-orientation
        scale = keypoints2[choice[1]][2] / keypoints1[choice[0]][2] #calculation first-scale ratio
        temp = []

        for j in range(len(matched_pairs)):
            if j is not rand:
                # calculation second-orientation
                orientation_temp = (keypoints1[matched_pairs[j][0]][3] - keypoints2[matched_pairs[j][1]][3]) % (2 * math.pi)
                # calculation second-scale ratio
                scale_temp = keypoints2[matched_pairs[j][1]][2] / keypoints1[matched_pairs[j][0]][2]
                # check degree error +- 30 degree
                if((orientation - math.pi/6) < orientation_temp) and (orientation_temp < (orientation + math.pi/6)): # -30 +30 범위 확인
                    # check scale error +- 50%
                    if(scale - scale*scale_agreement < scale_temp and scale_temp < scale + scale*scale_agreement):
                        temp.append([i, j])
        if(len(temp) > len(largest_set)): # choice best match
            largest_set = temp
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0], matched_pairs[largest_set[i][1]][1])

    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    y1 = descriptors1.shape[0] # descriptors1 size
    y2 = descriptors2.shape[0] # descriptors2 size
    # des1, des2 크기 저장
    temp = np.zeros(y2) # make an array of descriptors2 size

    # 일치하는 쌍을 찾기 위해 리스트 생성
    matched_pairs = []

    for i in range(y1):
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i],descriptors2[j])) #calculate the number of all cases

        compare = sorted(range(len(temp)), key = lambda k : temp[k]) # comparison to find the best
        # temp 배열 값을 오름차순 정렬

        if (temp[compare[0]]/temp[compare[1]]) < threshold: #check the best match
            matched_pairs.append([i, compare[0]])

        # 첫번째 값, 두번째 값 비율이 threshold보다 작으면 해당 pair을 좋은 매치로 판단

    #num = 5
    #matched_pairs = [[i, i] for i in range(num)]

    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START

    # 키포인트 개수 계산
    point_count = xy_points.shape[0]

    # 결과 배열 초기화
    xy_points_out = np.zeros((point_count, 2), dtype=np.float32)

    # 각 points h 행렬 곱해서 projection 계산
    for i in range(point_count):
        # 행렬 연산을 위해 1추가해서 demention 추가
        coordinate = np.append(xy_points[i], np.array([1]))

        # h행렬을 곱해 계산
        projected_point = np.dot(h, np.reshape(coordinate, (3, 1)))

        # z 좌표 값
        z = projected_point[2, 0]

        # z = 0 이면 z=1e-10으로 변경 devide zero 에러 방지
        if (z == 0):
            z = 1e-10

        # 투영된 점의 좌표를 z로 나눠줌, 수업시간에 배움
        projected_point /= z

        # z로 나눈 점 x, y좌표를 결과 배열에 저장
        xy_points_out[i] = projected_point[:2, 0]

    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    # 입력값 검증하는 코드
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START

    #변수 초기화
    num_matches = xy_src.shape[0]
    h = np.zeros((3, 3), dtype=np.float32)

    # 계산의 편의를 위해 xy_src, xy_ref를 합친다 xy_set.
    xy_set = np.hstack((xy_src, xy_ref))

    # 가장 큰 합의(consensus) 집합을 저장할 리스트 초기화
    largest_consenses_set = []

    # RANSAC 알고리즘 반복 수행
    for i in range(num_iter):
        # 중복없이 랜덤하게 4개의 매칭쌍을 고른다.
        matches = xy_set[np.random.choice(num_matches, 4)]

        # homography 행렬
        A = np.zeros((8, 9), dtype=np.float32)


        for j in range(4):
            src_point = np.append(matches[j, :2], np.array([1]))
            ref_point = np.append(matches[j, 2:], np.array([1]))
            # A 행렬을 구한다.
            A[j * 2] = np.hstack((src_point, np.array([0, 0, 0]), src_point * -1 * ref_point[0]))
            A[j * 2 + 1] = np.hstack((np.array([0, 0, 0]), src_point, src_point * -1 * ref_point[1]))

        # 라이브러리를 사용해 A의 고유값, 고유벡터를 구한다.
        A_eigenvalue, A_eigenvector = np.linalg.eig(np.dot(np.transpose(A), A))

        # 최소고유값을 구하기 위해 인덱스 정렬한다.
        idx = np.argsort(np.array(A_eigenvalue))

        A_eigenvalue = A_eigenvalue[idx]
        A_eigenvector = A_eigenvector[:, idx]

        # 최소 고유벡터로부터 homography 매트릭스 candidate 구함
        h_candidate = np.reshape(A_eigenvector[:, 0], (3, 3))

        consenses_set = []

        # xy_set의 각 매칭 쌍에 수행
        for match in xy_set:
            src_point_coordinate = np.append(match[:2], np.array([1]))
            src_point_coordinate = np.reshape(src_point_coordinate, (3, 1))

            # 각 점들을 projection시킨다.
            projected_src = np.dot(h_candidate, src_point_coordinate)
            z = projected_src[2]
            if (z == 0):
                z = 1e-10
            projected_src /= z
            ref_point = match[2:]

            # 유클리디안 거리를 구한다.
            distance = math.sqrt((ref_point[0] - projected_src[0]) ** 2 + (ref_point[1] - projected_src[1]) ** 2)

            # 거리가 tol보다 작으면 outlier로 취급하고, 아니면 inlier이다.
            if (distance < tol):
                consenses_set.append(match)

        # 가장 큰 subset만을 남긴다.
        if (len(consenses_set) > len(largest_consenses_set)):
            largest_consenses_set = consenses_set

    # 가장 큰 subset의 h 행렬이 final이다.
    num_largest = len(largest_consenses_set)
    largest_consenses_set = np.array(largest_consenses_set)
    matches = largest_consenses_set[np.random.choice(num_largest, 4)]

    # 똑같이 A 행렬을 구하고, 최종적으로 h 행렬을 구하면 끝이다.
    A = np.zeros((8, 9), dtype=np.float32)
    for j in range(4):
        src_point = np.append(matches[j, :2], np.array([1]))
        ref_point = np.append(matches[j, 2:], np.array([1]))
        A[j * 2] = np.hstack((src_point, np.array([0, 0, 0]), src_point * -1 * ref_point[0]))
        A[j * 2 + 1] = np.hstack((np.array([0, 0, 0]), src_point, src_point * -1 * ref_point[1]))

    A_eigenvalue, A_eigenvector = np.linalg.eig(np.dot(np.transpose(A), A))
    idx = np.argsort(np.array(A_eigenvalue))
    A_eigenvalue = A_eigenvalue[idx]
    A_eigenvector = A_eigenvector[:, idx]
    h = np.reshape(A_eigenvector[:, 0], (3, 3))

    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
