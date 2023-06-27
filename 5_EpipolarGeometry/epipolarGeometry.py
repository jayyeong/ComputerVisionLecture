import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    ### YOUR CODE BEGINS HERE
    # Af = 0 을 만들, minimize |Af|, AtA의 eigenvector

    A = np.zeros((n,9), dtype = np.float32) #여기서는 8*9 말고 n*9

    # u1u1' v1u1' u1' u1v1' v1v1' v1' u1 v1 1

    for i in range(n):
        pass
        temp_x1 = np.reshape(x1[:,i], (1,3))
        temp_x2 = np.reshape(x2[:,i], (3,1))
        temp_matrix = temp_x2 @ temp_x1 # 3 by 3
        A[i] = np.reshape(temp_matrix, (1, 9))

    # build matrix for equations in Page 51

    eigen_value, eigen_vector = np.linalg.eig(A.T @ A)

    least_index = np.argmin(eigen_value) #minimize하는 index

    F = np.reshape(eigen_vector[:,least_index], (3, 3))
    F = F/F[2][2] #normalization

    # compute the solution in Page 51

    U, sigma, V_transpose = np.linalg.svd(F) # 특이값 분해

    sigma[2] = 0 # rank 2
    sigma_prime = np.diag(sigma) # 대각행렬 만들기

    # constrain F: make rank 2 by zeroing out last singular value (Page 52)

    F = U @ sigma_prime @ V_transpose
    ### YOUR CODE ENDS HERE
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    U, S, V = np.linalg.svd(F) # solve F e1 = 0 by svd
    e1 = V[-1]
    e1 =  e1/e1[2] #normalization

    U, S, V = np.linalg.svd(F.T)  # solve FT e2 = 0 by svd
    e2 = V[-1]
    e2 = e2/e2[2] #normalization
    ### YOUR CODE ENDS HERE
    
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE

    point1 = cor1[:2,:].T
    # print(point1)
    point2 = cor2[:2,:].T
    n = cor1.shape[1]
    #print(n)

    height, width = img1.shape[0], img1.shape[1] #기본 가로세로길이 저장

    x = np.linspace(0, width, 10) #직선을 그리기 위해 x값 선언
    # print(x)

    graph = plt.figure()

    ax1 = graph.add_subplot(1, 2, 1)
    ax1.imshow(img1) #첫번째 이미지 추가
    ax1.scatter(cor1[0,:], cor1[1,:], s = 10) #epipole 삽입

    for i in range(n): #직선의 방정식
        y = (point1[i, 1] - e1[1]) / (point1[i, 0] - e1[0]) * (x - point1[i, 0]) + point1[i, 1]
        ax1.plot(x, y, linewidth = 1)

    ax1.set_xlim([0, width])
    ax1.set_ylim([height, 0]) #크기 원상복귀

    # 두번째 이미지도 같은 방식으로 수행
    ax2 = graph.add_subplot(1, 2, 2)
    ax2.imshow(img2) #두번째 이미지 추가
    ax2.scatter(cor2[0, :], cor2[1, :], s=10)

    for i in range(n): #직선의 방정식
        y = (point2[i, 1] - e2[1]) / (point2[i, 0] - e2[0]) * (x - point2[i, 0]) + point2[i, 1]
        ax2.plot(x, y, linewidth = 1)

    ax2.set_xlim([0, width])
    ax2.set_ylim([height, 0]) #크기 원상복귀

    plt.show()

    ### YOUR CODE ENDS HERE
    return

draw_epipolar_lines(img1, img2, cor1, cor2)