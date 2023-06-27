from PIL import Image
import numpy as np
import math

# 합이 1인 n * n 리스트 리턴
def boxfilter(n):
    assert n % 2 == 1, 'Dimension must be odd'
    # n이 홀수가 아니면, error출력
    return np.full((n, n), 1/(n*n))

# boxfilter(3)
# #boxfilter(4)
# boxfilter(7)

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

# gauss1d(0.3)
# gauss1d(0.5)
# gauss1d(1)
# gauss1d(2)

# 합이 1인 가우스 2차원 행렬
def gauss2d(sigma):
    x = gauss1d(sigma)
    x = np.outer(x, x)
    x = x / x.sum()

    return x

# gauss2d(0.5)
# gauss2d(1)

#이미지 convolution
def convolve2d(array, filter):
    array.astype(np.float32)
    filter.astype(np.float32)

    filter = np.flip(filter)
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
def gaussconvolve2d(array, sigma):
    filter = gauss2d(sigma)
    return  convolve2d(array, filter)

im = Image.open('2b_dog.bmp') # 이미지 불러오기
im_aa = np.asarray(im)
im.show() # 원본 이미지 출력
im = im.convert('L') #greyscale 변환
im_array = np.asarray(im)
im_array_con = gaussconvolve2d(im_array, 3) # sigma = 3
im_array_con = im_array_con.astype('uint8') #실수에서 uint8 변환
im_con = Image.fromarray(im_array_con) # 이미지로 변환
im_con.show() # 변환된 이미지 출력

# part2
def gaussconvolve2d_RGB(array, sigma):
    # r,g,b 로 각각 분리후 가우시안 컨볼루션 실행
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    # 각각 convolution
    low_r = gaussconvolve2d(red, sigma)
    low_g = gaussconvolve2d(green, sigma)
    low_b = gaussconvolve2d(blue, sigma)
    # np.dstack() 함수를 사용해 합쳐준다.
    im_low_array = np.dstack((low_r, low_g, low_b))
    im_low_array = np.array(im_low_array)
    return im_low_array

# part 2.1
eiffel = Image.open('3a_eiffel.bmp') # 이미지 불러오기
e_array = np.asarray(eiffel)
e_low_array = gaussconvolve2d_RGB(e_array, 3)
e_low_array = np.array(e_low_array, dtype= np.uint8)
e_low = Image.fromarray(e_low_array)
e_low.show()

# part 2.2
tower = Image.open('3b_tower.bmp')
t_array = np.asarray(tower)
t_low_array = gaussconvolve2d_RGB(t_array, 3)
# 이미지가 잘 보이도록 128 더해준다.
t_high_array = t_array - t_low_array + 128
# 0 ~ 255범위 넘어가는 값 처리
t_high_array = np.where(t_high_array < 0.0 , 0.0, t_high_array)
t_high_array = np.where(t_high_array > 255.0, 255.0, t_high_array)
t_high_array = np.array(t_high_array, dtype=np.uint8)
t_high = Image.fromarray(t_high_array)
t_high.show()

# part 2.3
eiffel = Image.open('3a_eiffel.bmp')
e_array = np.asarray(eiffel)
e_low_array = gaussconvolve2d_RGB(e_array, 3) # low frequency 추출
tower = Image.open('3b_tower.bmp')
t_array = np.asarray(tower)
t_low_array = gaussconvolve2d_RGB(t_array, 3)
t_high_array = t_array - t_low_array # high frequency 추출
# 이미지 합치기
hibrid_array = e_low_array + t_high_array
# 0 ~ 255범위 넘어가는 값 처리
hibrid_array = np.where(hibrid_array < 0., 0., hibrid_array)
hibrid_array = np.where(hibrid_array > 255., 255., hibrid_array)
hibrid_array = np.array(hibrid_array, dtype= np.uint8)
hibrid = Image.fromarray(hibrid_array)
hibrid.show() # 하이브리드 이미지 출력