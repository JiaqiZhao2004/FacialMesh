import cv2
import numpy as np
import dlib
from utils import extract_index


def initialize():
    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture("roy.mov")

    if not vid_capture.isOpened():
        print("Error opening the video file")
    # Read fps and frame count
    else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = vid_capture.get(5)
        print('Frames per second : ', fps, 'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    orientation_token_set = []
    indices_triangles_set = []

    while vid_capture.isOpened():
        ret, img = vid_capture.read()
        if ret:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(img_gray)
            for face in faces:
                landmarks = predictor(img_gray, face)
                landmarks_points = []

                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x, y))
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

                # convex hull
                points = np.array(landmarks_points, np.int32)

                # orientation token
                orientation_token = np.concatenate([points[:, 0] - points[30, 0], points[:, 1] - points[30, 1]])
                orientation_token = (orientation_token - np.average(orientation_token)) / np.std(orientation_token)

                convex_hull = cv2.convexHull(points)

                # Delaunay Triangulation
                rect = cv2.boundingRect(convex_hull)
                sub_div = cv2.Subdiv2D(rect)
                sub_div.insert(landmarks_points)
                triangles = sub_div.getTriangleList()
                triangles = np.array(triangles, np.int32)

                # indices of triangles
                indices_triangles = []
                for t in triangles:
                    point1 = (t[0], t[1])
                    point2 = (t[2], t[3])
                    point3 = (t[4], t[5])
                    # cv2.line(img, point1, point2, (0, 0, 255), 1)
                    # cv2.line(img, point3, point2, (0, 0, 255), 1)
                    # cv2.line(img, point1, point3, (0, 0, 255), 1)

                    index_point1 = extract_index(np.where((points == point1).all(axis=1)))
                    index_point2 = extract_index(np.where((points == point2).all(axis=1)))
                    index_point3 = extract_index(np.where((points == point3).all(axis=1)))

                    if index_point1 is not None and index_point2 is not None and index_point3 is not None:
                        indices_triangles.append((index_point1, index_point2, index_point3, point1, point2, point3))

                orientation_token_set.append(orientation_token)
                indices_triangles_set.append(indices_triangles)
                cv2.putText(img, "recorded", (100, 100), 1, 3, (0, 0, 0))
                break
            cv2.imshow('Frame', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()
    print(orientation_token_set)
    print(indices_triangles_set)
    return orientation_token_set, indices_triangles_set
