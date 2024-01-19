import cv2
import numpy as np
import dlib
from utils import extract_index


# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(0)

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

while vid_capture.isOpened():
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is a frame

    ret, img = vid_capture.read()
    if ret:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(img_gray)

        faces = detector(img_gray)
        landmarks_points = None
        indices_triangles = None
        face_clipped = None
        for face in faces:
            landmarks = predictor(img_gray, face)
            # corners = str(face)[1:-1].replace("(", "").replace(")", "").replace(",", "").split(" ")
            # landmarks_points = [(int(corners[0]), int(corners[1])), (int(corners[0]), int(corners[3])), (int(corners[2]), int(corners[1])),
            #                     (int(corners[2]), int(corners[3]))]
            landmarks_points = []

            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

            # convex hull
            points = np.array(landmarks_points, np.int32)

            convex_hull = cv2.convexHull(points)
            cv2.polylines(img, [convex_hull], True, (255, 0, 0), 1)

            # mask
            cv2.fillConvexPoly(mask, convex_hull, (255, 255, 255))

            # mask intersection with image
            face_clipped = cv2.bitwise_and(img, img, mask=mask)

            # Delaunay Triangulation
            rect = cv2.boundingRect(convex_hull)
            (x, y, w, h) = rect
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
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
                    indices_triangles.append((index_point1, index_point2, index_point3))
        cv2.imshow('Frame', img)
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(30)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()
