import os
from ultralytics import YOLO
import cv2
import math
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel but
    NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black."""
    #defining a blank mask to start with
    mask = np.zeros_like(img)                        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2: 
        channel_count = img.shape[2]                 
        ignore_mask_color = (255,) * channel_count   # 3 channel color mask
    else:
        ignore_mask_color = 255                      # 1 channel color mask
    # Fill the polygon defined by "vertices".  
    cv2.fillPoly(mask, vertices, ignore_mask_color)  # fill with RED color
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """`img` should be the output of a Canny transform.    
    Returns an image with hough lines drawn."""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # by using the dimensions of original image, creating a complete Blacked out copy of it.
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) 
    # calling the draw_lines function, which will draws lines on
    # the hough returned coordinates in fully Blacked-out Image.
    draw_lines(line_img, lines)                                             
    # this image has connected(line drawn) coordinates in complete Black-out image. 
    return line_img
    
def draw_lines(img, lines, color=[255,0,0], thickness=2):
    verticle_lines=[]   #  m=infinity
    horizontal_lines=[] #  m=0
    left_lines=[]   #  m=+ve
    right_lines=[]   #  m=-ve
    postiveSlope=0     #  +veSlopeSUM
    negtiveSlope=0     #  -veSlopeSUM
#     Seperate Left lines & Right Lines
    if lines is None:
        return
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (y2-y1)/(x2-x1) < 0.2 and (y2-y1)/(x2-x1) > -0.8:
                left_lines.append(line)
                postiveSlope+= ((y2-y1)/(x2-x1)) 
            elif (y2-y1)/(x2-x1) > 0.2 and (y2-y1)/(x2-x1) < 0.8:
                right_lines.append(line)
                negtiveSlope+= abs((y2-y1)/(x2-x1))                
                
#    coordinates average left_lanes , right_lanes
    LL_avg= np.average(left_lines, axis=0)
#     print('LL_avg',LL_avg)
    RL_avg= np.average(right_lines, axis=0)
#     print('RL_avg',RL_avg)
#    calculate the slope_avg & Intercept_avg for left_lanes , right_lanes

    for x1_avg,y1_avg,x2_avg,y2_avg in line:
        x1=x1_avg
        y1=y1_avg
        x2=x2_avg
        y2=y2_avg
        LL_slope_avg     = (y2 - y1)/(x2 - x1)+1
        LL_Intercept_avg = y1 - (LL_slope_avg * x1)
#     for x1_avg,y1_avg,x2_avg,y2_avg in RL_avg:
        x1=x1_avg
        y1=y1_avg
        x2=x2_avg
        y2=y2_avg
        RL_slope_avg     = (y2 - y1)/(x2 - x1)-1
        RL_Intercept_avg = y1 - (RL_slope_avg * x1)
    
    if len(left_lines)==0 or len(right_lines)==0:
        return
#    Calc the Lowest coordinate for left_lanes & right_lanes using x = (y - b)/m
    # min_left_y  =  min(y for xyxy in left_lines for y in xyxy[1::2])#find_minimum_y(left_lines)
    # min_left_x  =  int((min_left_y - LL_Intercept_avg)//LL_slope_avg)#calculate_x(min_left_y, LL_Intercept_avg, LL_slope_avg) #affecting right line
    
    # min_right_y =  min(y for xyxy in right_lines for y in xyxy[1::2])#find_minimum_y(right_lines)
    # min_right_x =  int((min_right_y - RL_Intercept_avg)//RL_slope_avg)#calculate_x(min_right_y, RL_Intercept_avg, RL_slope_avg) # affecting left line..?
    
    min_left_x, min_right_x, max_left_x, max_right_x = 64,64,64,64

    min_left_y = min(y for line in left_lines for y in line[0][1::2])
    max_left_y  =  max(y for line in left_lines for y in line[0][1::2])#find_maximum_y(left_lines)
    if LL_slope_avg != None and LL_slope_avg != 0:
        min_left_x = (min_left_y - LL_Intercept_avg) // LL_slope_avg
        max_left_x  =  (max_left_y - LL_Intercept_avg)// LL_slope_avg
        if math.isnan(min_left_x)==False:
            min_left_x = int(min_left_x)
        if math.isnan(max_left_x)==False:
            max_left_x = int(max_left_x)


    min_right_y = min(y for line in right_lines for y in line[0][1::2])
    max_right_y =  max(y for line in right_lines for y in line[0][1::2])#find_maximum_y(right_lines)    
    if RL_slope_avg != None and RL_slope_avg != 0:
        min_right_x = (min_right_y - RL_Intercept_avg) // RL_slope_avg
        max_right_x =  (max_right_y - RL_Intercept_avg)// RL_slope_avg 
        if not math.isnan(min_right_x):
            min_right_x = int(min_right_x)
        if not math.isnan(max_right_x):
            max_right_x = int(max_right_x)


#    Calc the highest coordinate for left_lanes & right_lanes using x = (y - b)/m
    #calculate_x(max_left_y,LL_Intercept_avg, LL_slope_avg) # affecting right line..?

    """This function draws `lines` with `color` and `thickness`."""

#   left_lane_lines drawn:  ==================  
    if isinstance(min_left_x, int) and isinstance(min_left_y, int) and isinstance(max_left_x, int) and isinstance(max_left_y, int):                                       
        cv2.line(img, (min_left_x, min_left_y), (max_left_x, max_left_y), [255,255,0], thickness=3)
#   right_lane_lines drawn  ================== 
    if isinstance(min_right_x, int) and isinstance(min_right_y, int) and isinstance(max_right_x, int) and isinstance(max_right_y, int): 
        cv2.line(img, (min_right_x, min_right_y), (max_right_x, max_right_y), [0,0,255], thickness=3)

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α, β, λ):
    """`img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
                    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!"""
    return cv2.addWeighted(img, α, initial_img, β, λ)

'''This function pipeline that takes in an input image and outputs annotated Lane-lines'''
def laneFindingPipeLine(img):
    '''Inorder to apply canny algorithm, We have to convert color image to gray 
    i.e a single channel image. Gray image helps us to clearly identify, 
    1. DARK/ BRIGHT/ GRAY(in between) pixels in the image. 
    2. Rapid changes in the brightess which are edges/dots detected by Canny Algo.'''
    gray_image = grayscale(img)

    '''filter-out noise and spurious gradients by averaging  with gaussian filter,
    keeping Kernel_size = 5, **always select a odd size number.'''
    kernel_size=3
    blur_gray = gaussian_blur(gray_image, kernel_size)

    '''now, we can apply canny algorithm, that apply its gaussian filter--> 
    finds the gradient strength & direction(you get thick edges)-->
    apply non-max. suppression(get thin/candidate edges)-->
    Hysteresis with the help of thresholds specifies in the arguments'''
    low_threshold  = 50
    high_threshold = 100
    canny_edges = canny(blur_gray, low_threshold, high_threshold)

    '''Now, you have a edge-detected image(SINGLE CHANNEL) : which has edges all over the image 
    based on you detection criterion(Arguments of canny_algorithm). This doesn't
    solve our purpose. As canny_edges are all over the image.
    We need to narrow-down our area of visibility to not to let our algorithm confuse.
    By creating a REGION OF INTEREST (a POLYGON around the lane lines).
    This function 
    1. takes the coordinates of the polygon and draws a polygon in the image,
    2. then fill it with a RED color <-- this becomes your mask (a specific area to look on.)
    3. then it performs bitwise AND with canny_edge_detected_image(which is full of edges everywhere) 
       AND mask_image(just a plane RED POLYGON SHAPE in the image(SINGLE COLOR-GRAY))
       --> gives a canny_edge_detected_masked_image(SINGLE COLOR-GRAY)
    So, now you have clear RED-HIGHLIGHTNED canny- edges....BEAUTIFULL..!!! :)
    Note: Creating a mask is independent of any operation'''
    vertices = np.array([[(0,539),(425, 340), (510, 320), (959,539)]], dtype=np.int32)
    masked_image = region_of_interest(blur_gray, vertices)

    '''Moreover, you want to create Mask on the canny_edges that's why you have 
    to do a bit-wise AND to get only the masked-canny_edges Image.
            "Issue": masked_image shape (540, 960, 3) and canny_edges shape (540, 960),
            above I have given blur_gray instead of img to make them of same shape.
    Ultimately, we need a SINGLE COLOR IMAGE(GRAY) as an input to hough_lines()
    that take these edges/pixels check their intersections in Hought space and 
    connect them in Image space'''
    masked_canny_edges = cv2.bitwise_and(masked_image, canny_edges)

    '''Work is not done YET..!! We just have dotted-lane-lines detected.
    So, we connect them..? How we know which dot to connect with which other dot,
    & how many dots to connect? ...TRICKY SITUATION...!!!
    Awesome Manipulation by hough, We have dots in our image space, 
    which equivalent to a line for each dot in Hough space. 
    And, collinear points in image space is equivalent to respective lines 
    passing through common intersection point/pixel in Hough space. 
    Also, in Hough space we have grids cells, which helps to catch all the edges/dots
    in images space which are not collinear but very much near to collinear points 
    by adjusting THRESHOLD parameter in below function. The more count we get of lines 
    passing through a grid pixel in hough space, the more we get 
    collinear/near-collinear edges/dots in out image space.
    This way we will be able to clear identify the sequence of coordinates of all 
    near-collinear points in the canny_edge_detected_masked_image ......AMAZING..!!
    
    Now, you have the coordinates, Connect them with a line, by calling custom draw_lines()'''
    rho = 1
    theta = np.pi/180
    threshold = 12
    min_line_len = 10
    max_line_gap = 2
    # --> will call draw lines & you get a Blacked-out image with connected Hough returned coordinates.
    line_img = hough_lines(masked_canny_edges, rho, theta, threshold, min_line_len, max_line_gap)

    '''Now we have
                1. a image with hough Annotated lines.
                2. a original image. 
    this function will give weights to each pixel & glue them together. 
    Remember, before gluing, the dimension should be same, else it pops out errors.'''
    output = weighted_img(line_img, img, α=0.5, β=1.0, λ=0.)
    return output

    # Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

''' Function that calls my pipeline on each frame of my video, Actually,a video is a sequence of images. :)'''
def process_image(image):  
    result = laneFindingPipeLine(image)   
    return result

white_output = '../output/checking_roads.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip('../output/video_ts_3_min.mp4')
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)