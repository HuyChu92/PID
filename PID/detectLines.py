import cv2 as cv
import numpy as np
import matplotlib.image as mpimg
import math
import os


def save_individual_lines(lines, image):
    # Create a directory to save individual lines if it doesn't exist
    if not os.path.exists('individual_lines'):
        os.makedirs('individual_lines')

    for i, line in enumerate(lines):
        leftx, boty, rightx, topy = line
        # Crop the line from the original image
        cropped_line = image[boty:topy, leftx:rightx]
        # Save the cropped line as an individual image
        line_filename = f'individual_lines/line_{i}.jpg'
        cv.imwrite(line_filename, cropped_line)

def get_lines(lines_in):
    if cv.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]


# Define a function to calculate the distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_lines(image_src, output_filename=None):
    # Define a threshold for considering lines as connected
    threshold_distance = 10  # Adjust this value as needed
    img = mpimg.imread(image_src)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    ret, thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    
    thresh1 = cv.bitwise_not(thresh1)
    
    edges = cv.Canny(thresh1, threshold1=50, threshold2=200, apertureSize = 3)

    lines = cv.HoughLinesP(thresh1, rho=1, theta=np.pi/180, threshold=10,
                            minLineLength=5, maxLineGap=30)

    # l[0] - line; l[1] - angle
    for line in get_lines(lines):
        leftx, boty, rightx, topy = line
        cv.line(img, (leftx, boty), (rightx,topy), (0,0,255), 6) 
        
    # merge lines
        
    #------------------
    # prepare
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
        
    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)
            
    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])
        
    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)
    
    unique_lines = []
    unique_lines.extend(merged_lines_x)
    unique_lines.extend(merged_lines_y)
    print("process groups lines", len(_lines), len(unique_lines))
    img_merged_lines = mpimg.imread(image_src)
    for line in unique_lines:
        cv.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)

    
    cv.imwrite('prediction/lines_gray.jpg',gray)
    cv.imwrite('prediction/lines_thresh.jpg',thresh1)
    cv.imwrite('prediction/lines_edges.jpg',edges)
    cv.imwrite('prediction/lines_lines.jpg',img)
    # cv.imwrite('merged_lines.jpg',img_merged_lines)
    cv.imwrite(output_filename,img_merged_lines)

    
    # Create a list to store connected line groups
    connected_line_groups = []
    # Iterate over each merged line
    for line in unique_lines:
        # Get the endpoints of the current line
        start_point = line[0]
        end_point = line[1]

        # Create a new group for the current line
        new_group = [line]

        # Iterate over the remaining merged lines to check for connections
        for other_line in unique_lines:
            if line == other_line:
                continue  # Skip checking against itself

            # Get the endpoints of the other line
            other_start_point = other_line[0]
            other_end_point = other_line[1]

            # Check if the start or end points of the lines are within the threshold distance
            if (
                distance(start_point, other_start_point) < threshold_distance
                or distance(start_point, other_end_point) < threshold_distance
                or distance(end_point, other_start_point) < threshold_distance
                or distance(end_point, other_end_point) < threshold_distance
            ):
                # If connected, add the other line to the current group
                new_group.append(other_line)

        # Add the group of connected lines to the list
        connected_line_groups.append(new_group)

    total_unique_lines = len(unique_lines)
    print(f"Total unique lines: {total_unique_lines}")
    return unique_lines

def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 30
    min_angle_to_merge = 30
    
    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break
            
            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)
        
    
    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))
    
    return super_lines_final

def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]
    
    line_i = lines[0]
    
    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
        
    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        
        #sort by y
        points = sorted(points, key=lambda point: point[1])
        
        if use_log:
            print("use y")
    else:
        
        #sort by x
        points = sorted(points, key=lambda point: point[0])
        
        if use_log:
            print("use x")
    
    return [points[0], points[len(points)-1]]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])
    
    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False
    
def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude
 
#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/
def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    
    
    return min(dist1,dist2,dist3,dist4)

# image_source = r'C:\Users\huy_c\Desktop\Projects\imageprocessing\output_image.jpg'
# process_lines(image_source)