from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""

# 합이 1인 1차원 가우스 행렬 리턴
def gauss1d(sigma):
    length = math.ceil(sigma * 6)
    if length % 2 == 0:
        length += 1

    center = length // 2
    x = np.arange(length) - center
    x = np.exp(-(x**2)/(2*sigma**2))
    x = x / x.sum()

    return x

# 합이 1인 가우스 2차원 행렬
def gauss2d(sigma):
    x = gauss1d(sigma)
    x = np.outer(x, x)
    x = x / x.sum()

    return x

#이미지 convolution
def convolve2d(array,filter):
    array.astype(np.float32)
    filter.astype(np.float32)

    #filter = np.flip(filter)
    filter = np.rot90(np.rot90(filter))
    # convolution을 위해 필터를 180도 회전(점 대칭)한다.

    a_length, a_width = array.shape
    f_length, f_width = filter.shape

    # padding 크기 설정
    pad_length = (f_length - 1) // 2
    pad_width = (f_width - 1) // 2

    padded_array = np.pad(array, ((pad_length, pad_length), (pad_width, pad_width)), mode='constant', constant_values = 0)
    # kernal 크기에 따라 zero padding 해준다.

    output = np.zeros((a_length, a_width))

    #convolution
    for i in range(a_length):
        for j in range(a_width):
            result = np.sum(padded_array[i:i + f_length, j:j + f_width] * filter)
            output[i][j] = result

    return output

# gaussian filter을 이용한 convolution
def gaussconvolve2d(array,sigma):
    filter = gauss2d(sigma)
    return convolve2d(array, filter)

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """

    # sobel kernels
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # 1/8
    x_filter /= 8.
    y_filter /= 8.

    # convolution x방향, y방향
    img_x = convolve2d(img, x_filter)
    img_y = convolve2d(img, y_filter)

    # magnitude with normalization
    img_xy = np.hypot(img_x, img_y)
    img_xy = img_xy/img_xy.max() * 255

    # 이미지 thresholding
    img_xy = np.where(img_xy < 0.0, 0.0, img_xy)
    img_xy = np.where(img_xy > 255.0, 255.0, img_xy)
    #img_xy = np.clip(img_xy, 0., 255.) # 위와 동일한 기능

    # magnitude
    G = img_xy
    # direction
    theta = np.arctan2(img_y, img_x)
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    degree = np.rad2deg(theta) # radian to degree

    H, W = G.shape

    res = np.zeros((H, W), dtype = np.uint8)

    # 9시가 0도, 12시가 90도, 3시가 180도
    for i in range(1, H - 1):
        for j in range(1, W - 1):

            # 0도
            if (0. <= degree[i, j] < 22.5) or (157.5 <= degree[i, j] <= 180.) or (-22.5 <= degree[i, j] < 0.) or (-180. <= degree[i, j] < -157.5):
                left = G[i, j - 1]
                right = G[i, j + 1]

            # 45도
            elif (22.5 <= degree[i, j] < 67.5) or (-157.5 <= degree[i, j] < -112.5):
                left = G[i - 1, j - 1]
                right = G[i + 1, j + 1]

            # 90도
            elif (67.5 <= degree[i, j] < 112.5) or (-112.5 <= degree[i, j] < -67.5):
                left = G[i - 1, j]
                right = G[i + 1, j]

            # 135도
            elif (112.5 <= degree[i, j] < 157.5) or (-67.5 <= degree[i, j] < -22.5):
                left = G[i - 1, j + 1]
                right = G[i + 1, j - 1]

            if (G[i, j] > left) and (G[i, j] > right):
                res[i, j] = G[i, j]

    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """

    #임계값 정하기
    diff = img.max() - img.min()
    T_high = img.min() + diff * 0.15
    T_low = img.min() + diff * 0.03

    #strong edge 175에서 +80 더해져서 255
    strong_img = np.where(img >= T_high, 175, 0)
    # weak edge 80
    weak_img = np.where(img >= T_low, 80, 0)

    #두 이미지 합치기
    res = np.uint8(strong_img + weak_img)
    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """

    H, W = img.shape

    res = np.where(img == 255, 255, 0)

    visit = []

    #이미지 탐색
    for i in range(H):
        for j in range(W):
            if img[i, j] == 255: #connect할 weak edge는 strong edge에 붙어있으니까
                dfs(img, res, i, j, visit)

    return res

# 0
iguana = Image.open('iguana.bmp') # 이미지 열기
iguana.show() # 이미지 출력

iguana_g = iguana.convert('L') # convert greyscale
iguana_g_array = np.asarray(iguana_g)
iguana_g_array = iguana_g_array.astype('float32')

# 1
iguana_g_array_con = gaussconvolve2d(iguana_g_array, 1.6) # gaussian convolution
iguana_g_con = iguana_g_array_con.astype('uint8')
iguana_g_con = Image.fromarray(iguana_g_array_con)
iguana_g_con.show()

# 2
iguana_g_array_con = iguana_g_array_con.astype('float32')
G_, theta_ = sobel_filters(iguana_g_array_con)
img_sobel = G_.astype('uint8')
img_sobel = Image.fromarray(img_sobel)
img_sobel.show()

# 3
nms_array = non_max_suppression(G_, theta_)
nms_img = nms_array.astype('uint8')
nms_img = Image.fromarray(nms_img)
nms_img.show()

#4
d_array = double_thresholding(nms_array)
d_img = d_array.astype('uint8')
d_img = Image.fromarray(d_img)
d_img.show()

#5
h_array = hysteresis(d_array)
h_img = h_array.astype('uint8')
h_img = Image.fromarray(h_img)
h_img.show()