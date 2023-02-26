import numpy as np
import cv2
from coordinates import Point, Line

def get_coordinates(shape, results):
    """
    get coordinates of the facial keypoints
    """
    face = results.multi_face_landmarks[0]

    coordinates = []
    for landmark in face.landmark:
        x = landmark.x
        y = landmark.y

        # convert relative coordinates into acutal image size
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])

        coordinates.append([relative_x, relative_y])

    return coordinates


def plot_keypoints(image, results, label_red=None):
    """
    Plot an image of the facial keypoints in blue.

    label_red is a list of indexes in canonical_face_model_uv_visualization.png
    """

    annotated_image = image.copy()
    coordinates = get_coordinates(image.shape, results)

    for point in coordinates:
        cv2.circle(annotated_image, point, radius=1, color=(255, 0, 100), thickness=2)

    if label_red:
        for index in label_red:
            cv2.circle(annotated_image, coordinates[index], radius=1, color=(0, 0, 255), thickness=3)

    return  annotated_image


def grid_4(image, results):
    annotated_image = image.copy()
    coordinates = get_coordinates(image.shape, results)

    bottom_nose = Point.np_array_to_Point( coordinates[2] )
    
    lip_left_edge = Point.np_array_to_Point( coordinates[61] )
    lip_right_edge = Point.np_array_to_Point( coordinates[291] )
    
    left_lip_top =  Point.np_array_to_Point( coordinates[37] )
    right_lip_top =  Point.np_array_to_Point( coordinates[267] )
    
    bottom_lip =  Point.np_array_to_Point( coordinates[17] )
    

    # top line
    cv2.line(annotated_image, bottom_nose.get(lip_left_edge.x), bottom_nose.get(lip_right_edge.x),(255,0,0),2)
    top_line = Line(bottom_nose.get_point(lip_left_edge.x), bottom_nose.get_point(lip_right_edge.x))

    # middle line
    middle_line = Line(left_lip_top, right_lip_top)
    cv2.line(annotated_image, (lip_left_edge.x, middle_line.solve(lip_left_edge.x)), (lip_right_edge.x, middle_line.solve(lip_right_edge.x)),(255,0,0),2)
    
    # bottom line
    cv2.line(annotated_image, bottom_lip.get(lip_left_edge.x), bottom_lip.get(lip_right_edge.x),(255,0,0),2)
    bottom_line = Line(bottom_lip.get_point(lip_left_edge.x), bottom_lip.get_point(lip_right_edge.x))

    
    # top left point
    top_left_point = (lip_left_edge.x, top_line.solve(lip_left_edge.x))

    # top right point
    top_right_point = (lip_right_edge.x, top_line.solve(lip_right_edge.x))

    # middle left point
    middle_left_point = (lip_left_edge.x, middle_line.solve(lip_left_edge.x))

    # middle right point
    middle_right_point = (lip_right_edge.x, middle_line.solve(lip_right_edge.x))

    # bottom left point
    bottom_left_point = (lip_left_edge.x, bottom_line.solve(lip_left_edge.x))

    # bottom right point
    bottom_right_point = (lip_right_edge.x, bottom_line.solve(lip_right_edge.x))

    # left horizontal line
    cv2.line(annotated_image, top_left_point, bottom_left_point,(255,0,0),2)

    # right horizontal line
    cv2.line(annotated_image, top_right_point, bottom_right_point,(255,0,0),2)

# #     # points
    cv2.circle(annotated_image, top_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, top_right_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, middle_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, middle_right_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, bottom_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, bottom_right_point, radius=1, color=(0, 0, 255), thickness=3)
    
    
#     print("top", top_right_point, "middle", middle_right_point, "bottom", bottom_right_point)
    top_distance = middle_right_point[1] - top_right_point[1]
    bottom_distance = bottom_right_point[1] - middle_right_point[1]
    print("ratio:", bottom_distance/top_distance)
    
#     # add text of ratio
    text_y = ((middle_right_point[1] + bottom_right_point[1]) / 2)
    
    cv2.putText(img=annotated_image, text=str(bottom_distance/top_distance), org=(top_right_point[0], int(text_y)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 173),thickness=2)
    
    return annotated_image


def grid_11(image, results):
    annotated_image = image.copy()
    coordinates = get_coordinates(image.shape, results)
    
    
    
    left_eyebrow_top =  Point.np_array_to_Point( coordinates[105] )
    right_eyebrow_top =  Point.np_array_to_Point( coordinates[334] )

    left_lip_top =  Point.np_array_to_Point( coordinates[37] )
    right_lip_top =  Point.np_array_to_Point( coordinates[267] )

    left_eyebrow_edge =  Point.np_array_to_Point( coordinates[156] )
    right_eyebrow_edge =  Point.np_array_to_Point( coordinates[300] )

    bottom_chin =  Point.np_array_to_Point( coordinates[152] )
    

    # top line
    cv2.line(annotated_image, left_eyebrow_top.get(left_eyebrow_edge.x), right_eyebrow_top.get(right_eyebrow_edge.x),(255,0,0),2)
    top_line = Line(left_eyebrow_top, right_eyebrow_top)

    # middle line
    middle_line = Line(left_lip_top, right_lip_top)
    cv2.line(annotated_image, (left_eyebrow_edge.x, middle_line.solve(left_eyebrow_edge.x)), (right_eyebrow_edge.x, middle_line.solve(right_eyebrow_edge.x)),(255,0,0),2)
    
    # bottom line
    cv2.line(annotated_image, bottom_chin.get(left_eyebrow_edge.x), bottom_chin.get(right_eyebrow_edge.x),(255,0,0),2)
    bottom_line = Line(bottom_chin.get_point(left_eyebrow_edge.x), bottom_chin.get_point(right_eyebrow_edge.x))

    
#     # top left point
    top_left_point = (left_eyebrow_edge.x, top_line.solve(left_eyebrow_edge.x))

#     # top right point
    top_right_point = (right_eyebrow_edge.x, top_line.solve(right_eyebrow_edge.x))

#     # middle left point
    middle_left_point = (left_eyebrow_edge.x, middle_line.solve(left_eyebrow_edge.x))

#     # middle right point
    middle_right_point = (right_eyebrow_edge.x, middle_line.solve(right_eyebrow_edge.x))

#     # bottom left point
    bottom_left_point = (left_eyebrow_edge.x, bottom_line.solve(left_eyebrow_edge.x))

#     # bottom right point
    bottom_right_point = (right_eyebrow_edge.x, bottom_line.solve(right_eyebrow_edge.x))

#     # left horizontal line
    cv2.line(annotated_image, top_left_point, bottom_left_point,(255,0,0),2)

#     # right horizontal line
    cv2.line(annotated_image, top_right_point, bottom_right_point,(255,0,0),2)

#     # points
    cv2.circle(annotated_image, top_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, top_right_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, middle_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, middle_right_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, bottom_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, bottom_right_point, radius=1, color=(0, 0, 255), thickness=3)
    
    
#     print("top", top_right_point, "middle", middle_right_point, "bottom", bottom_right_point)
    top_distance = middle_right_point[1] - top_right_point[1]
    bottom_distance = bottom_right_point[1] - middle_right_point[1]
    print("ratio:", top_distance/bottom_distance)
    
#     # add text of ratio
    text_y = ((top_right_point[1] + middle_right_point[1]) / 2)
    
    cv2.putText(img=annotated_image, text=str(top_distance/bottom_distance), org=(top_right_point[0], int(text_y)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 173),thickness=2)
    
    return annotated_image


def grid_15(image, results):
    annotated_image = image.copy()
    coordinates = get_coordinates(image.shape, results)
    
    
    
    left_eyebrow_top =  Point.np_array_to_Point( coordinates[105] )
    right_eyebrow_top =  Point.np_array_to_Point( coordinates[334] )

    left_eye_top =  Point.np_array_to_Point( coordinates[159] )
    right_eye_top =  Point.np_array_to_Point( coordinates[386] )

    left_eyebrow_edge =  Point.np_array_to_Point( coordinates[156] )
    right_eyebrow_edge =  Point.np_array_to_Point( coordinates[300] )

    left_eye_bottom =  Point.np_array_to_Point( coordinates[145] )
    right_eye_bottom =  Point.np_array_to_Point( coordinates[374] )    
    

    # top line
    cv2.line(annotated_image, left_eyebrow_top.get(left_eyebrow_edge.x), right_eyebrow_top.get(right_eyebrow_edge.x),(255,0,0),2)
    top_line = Line(left_eyebrow_top, right_eyebrow_top)

    # middle line
    cv2.line(annotated_image, left_eye_top.get(left_eyebrow_edge.x), right_eye_top.get(right_eyebrow_edge.x),(255,0,0),2)
    middle_line = Line(left_eye_top, right_eye_top)

    # bottom line
    cv2.line(annotated_image, left_eye_bottom.get(left_eyebrow_edge.x), right_eye_bottom.get(right_eyebrow_edge.x),(255,0,0),2)
    bottom_line = Line(left_eye_bottom, right_eye_bottom)

    # top left point
    top_left_point = (left_eyebrow_edge.x, top_line.solve(left_eyebrow_edge.x))

    # top right point
    top_right_point = (right_eyebrow_edge.x, top_line.solve(right_eyebrow_edge.x))

    # middle left point
    middle_left_point = (left_eyebrow_edge.x, middle_line.solve(left_eyebrow_edge.x))

    # middle right point
    middle_right_point = (right_eyebrow_edge.x, middle_line.solve(right_eyebrow_edge.x))

    # bottom left point
    bottom_left_point = (left_eyebrow_edge.x, bottom_line.solve(left_eyebrow_edge.x))

    # bottom right point
    bottom_right_point = (right_eyebrow_edge.x, bottom_line.solve(right_eyebrow_edge.x))

    # left horizontal line
    cv2.line(annotated_image, top_left_point, bottom_left_point,(255,0,0),2)

    # right horizontal line
    cv2.line(annotated_image, top_right_point, bottom_right_point,(255,0,0),2)

    # points
    cv2.circle(annotated_image, top_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, top_right_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, middle_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, middle_right_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, bottom_left_point, radius=1, color=(0, 0, 255), thickness=3)
    cv2.circle(annotated_image, bottom_right_point, radius=1, color=(0, 0, 255), thickness=3)
    
    
    print("top", top_right_point, "middle", middle_right_point, "bottom", bottom_right_point)
    top_distance = middle_right_point[1] - top_right_point[1]
    bottom_distance = bottom_right_point[1] - middle_right_point[1]
    print("ratio:", top_distance/bottom_distance)
    
    # add text of ratio
    text_y = ((top_right_point[1] + middle_right_point[1]) / 2)
    
    cv2.putText(img=annotated_image, text=str(top_distance/bottom_distance), org=(top_right_point[0], int(text_y)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 173),thickness=2)
    
    return annotated_image