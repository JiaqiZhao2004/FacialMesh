# color
# size
# rotation

# detecting landmark

import cv2
import numpy as np
import dlib

# 0: webcam
# 1: photo
# 2: video
mode = 0

ORIGINAL = "Front.png"
TARGET = "IMG_2259.jpg"


def extract_index(array):
    index = None
    for number in array[0]:
        index = number
        break
    return index


img = cv2.imread(ORIGINAL)  # original
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(img_gray)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
        # cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # convex hull
    points = np.array(landmarks_points, np.int32)
    convex_hull = cv2.convexHull(points)
    # cv2.polylines(img, [convex_hull], True, (255, 0, 0), 1)

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

assert (len(indices_triangles) > 0), "No face detected"

if mode == 1:
    target = cv2.imread(TARGET)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    final = np.zeros_like(target)
    target_faces = detector(target_gray)
    for face in target_faces:
        # target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(target_gray, face)
        target_landmarks_points = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            target_landmarks_points.append((x, y))
            # cv2.circle(target, (x, y), 5, (0, 0, 255), -1)

        # convex hull
        target_points = np.array(target_landmarks_points, np.int32)
        target_convex_hull = cv2.convexHull(target_points)
        (target_x, target_y, target_w, target_h) = cv2.boundingRect(target_convex_hull)
        # cv2.rectangle(target, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 1)
        # cv2.polylines(target, [target_convex_hull], True, (255, 0, 0), 3)

        # replace triangles from original to target
        for i in indices_triangles:
            # original
            tr1_pt1 = landmarks_points[i[0]]
            tr1_pt2 = landmarks_points[i[1]]
            tr1_pt3 = landmarks_points[i[2]]
            tr1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            (x, y, w, h) = cv2.boundingRect(tr1)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cropped_tr1 = img[y: y + h, x: x + w]
            cropped_tr1_gray = cv2.cvtColor(cropped_tr1, cv2.COLOR_BGR2GRAY)
            cropped_tr1_mask = np.zeros_like(cropped_tr1_gray)
            points1 = [[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]]
            cv2.fillConvexPoly(cropped_tr1_mask, np.array(points1, np.int32), (255, 255, 255))
            # pad = max(1, (h + w) // 15)
            # color = (255, 255, 255)
            # cv2.line(cropped_tr1_mask, points1[0], points1[1], color, pad)
            # cv2.line(cropped_tr1_mask, points1[1], points1[2], color, pad)
            # cv2.line(cropped_tr1_mask, points1[2], points1[0], color, pad)
            cropped_tr1 = cv2.bitwise_and(cropped_tr1, cropped_tr1, mask=cropped_tr1_mask)

            # target
            tr2_pt1 = target_landmarks_points[i[0]]
            tr2_pt2 = target_landmarks_points[i[1]]
            tr2_pt3 = target_landmarks_points[i[2]]
            # cv2.line(target, tr2_pt1, tr2_pt2, (0, 0, 255), 1)
            # cv2.line(target, tr2_pt1, tr2_pt3, (0, 0, 255), 1)
            # cv2.line(target, tr2_pt2, tr2_pt3, (0, 0, 255), 1)
            tr2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            (x, y, w, h) = cv2.boundingRect(tr2)
            # cv2.rectangle(target, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cropped_tr2 = target[y: y + h, x: x + w]
            cropped_tr2_mask = np.zeros((h, w), np.int8)
            points2 = [[tr2_pt1[0] - x, tr2_pt1[1] - y],
                       [tr2_pt2[0] - x, tr2_pt2[1] - y],
                       [tr2_pt3[0] - x, tr2_pt3[1] - y]]
            cv2.fillConvexPoly(cropped_tr2_mask, np.array(points2, np.int32), (255, 255, 255))
            cropped_tr2 = cv2.bitwise_and(cropped_tr2, cropped_tr2, mask=cropped_tr2_mask)

            # warp triangles (affine)
            points1 = np.float32(points1)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points1, points2)
            warped_tr = cv2.warpAffine(cropped_tr1, M, (w, h))

            # reconstruct final face
            triangle_area = final[y: y + h, x: x + w]
            for i in range(h):
                for j in range(w):
                    for c in range(3):
                        final[i + y, j + x, c] = max(final[i + y, j + x, c], warped_tr[i, j, c])

        # final = cv2.add(final, target)

        mask_target = np.zeros_like(final)
        cv2.fillConvexPoly(mask_target, target_convex_hull, (255, 255, 255))
        p = (target_x + target_w // 2, target_y + target_h // 2)
        final = cv2.seamlessClone(final, target, mask_target, p, 1)

        # fill face
        # corners = str(face)[1:-1].replace("(", "").replace(")", "").replace(",", "").split(" ")
        # [(int(corners[0]), int(corners[1])), (int(corners[0]), int(corners[3])), (int(corners[2]), int(corners[1])),
        #                     (int(corners[2]), int(corners[3]))]
        # cv2.rectangle(target, (int(corners[0]), int(corners[1])), (int(corners[2]), int(corners[3])), (0, 255, 0), -1)

    # cv2.imshow("Image", img)
    cv2.imshow("target", target)
    # cv2.imshow("Face Clipped", face_clipped)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Triangle 1", cropped_tr1)
    # cv2.imshow("Triangle 1 mask", cropped_tr1_mask)
    # cv2.imshow("Triangle 2", cropped_tr2)
    # cv2.imshow("Triangle 2 mask", cropped_tr2_mask)
    # cv2.imshow("Warped Triangle", warped_tr)
    cv2.imshow("final", final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if mode == 0:
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

    while vid_capture.isOpened():
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, target = vid_capture.read()
        if ret:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            final = np.zeros_like(target)
            target_faces = detector(target_gray)
            if len(target_faces) > 0:
                for face in target_faces:
                    landmarks = predictor(target_gray, face)
                    target_landmarks_points = []
                    for n in range(68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        target_landmarks_points.append((x, y))

                    # convex hull
                    target_points = np.array(target_landmarks_points, np.int32)
                    target_convex_hull = cv2.convexHull(target_points)
                    (target_x, target_y, target_w, target_h) = cv2.boundingRect(target_convex_hull)

                    # replace triangles from original to target
                    for i in indices_triangles:
                        # original
                        tr1_pt1 = landmarks_points[i[0]]
                        tr1_pt2 = landmarks_points[i[1]]
                        tr1_pt3 = landmarks_points[i[2]]
                        tr1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
                        (x, y, w, h) = cv2.boundingRect(tr1)

                        cropped_tr1 = img[y: y + h, x: x + w]
                        cropped_tr1_gray = cv2.cvtColor(cropped_tr1, cv2.COLOR_BGR2GRAY)
                        cropped_tr1_mask = np.zeros_like(cropped_tr1_gray)
                        points1 = [[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]]
                        cv2.fillConvexPoly(cropped_tr1_mask, np.array(points1, np.int32), (255, 255, 255))
                        cropped_tr1 = cv2.bitwise_and(cropped_tr1, cropped_tr1, mask=cropped_tr1_mask)

                        # target
                        tr2_pt1 = target_landmarks_points[i[0]]
                        tr2_pt2 = target_landmarks_points[i[1]]
                        tr2_pt3 = target_landmarks_points[i[2]]
                        tr2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
                        (x, y, w, h) = cv2.boundingRect(tr2)

                        cropped_tr2 = target[y: y + h, x: x + w]
                        cropped_tr2_mask = np.zeros((h, w), np.int8)
                        points2 = [[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                   [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                   [tr2_pt3[0] - x, tr2_pt3[1] - y]]
                        cv2.fillConvexPoly(cropped_tr2_mask, np.array(points2, np.int32), (255, 255, 255))
                        cropped_tr2 = cv2.bitwise_and(cropped_tr2, cropped_tr2, mask=cropped_tr2_mask)

                        # warp triangles (affine)
                        points1 = np.float32(points1)
                        points2 = np.float32(points2)
                        M = cv2.getAffineTransform(points1, points2)
                        warped_tr = cv2.warpAffine(cropped_tr1, M, (w, h))

                        # reconstruct final face
                        triangle_area = final[y: y + h, x: x + w]
                        for i in range(h):
                            for j in range(w):
                                for c in range(3):
                                    final[i + y, j + x, c] = max(final[i + y, j + x, c], warped_tr[i, j, c])

                    mask_target = np.zeros_like(target)
                    cv2.fillConvexPoly(mask_target, target_convex_hull, (255, 255, 255))
                    p = (target_x + target_w // 2, target_y + target_h // 2)
                    final = cv2.seamlessClone(final, target, mask_target, p, 1)
                cv2.imshow('Frame', final)
            else:
                cv2.imshow('Frame', target)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(30)

            if key == ord('q'):
                break
        else:
            break

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()

if mode == 2:
    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(TARGET)

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

    while vid_capture.isOpened():
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, target = vid_capture.read()
        if ret:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            final = np.zeros_like(target)
            target_faces = detector(target_gray)
            if len(target_faces) > 0:
                for face in target_faces:
                    landmarks = predictor(target_gray, face)
                    target_landmarks_points = []
                    for n in range(68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        target_landmarks_points.append((x, y))

                    # convex hull
                    target_points = np.array(target_landmarks_points, np.int32)
                    target_convex_hull = cv2.convexHull(target_points)
                    (target_x, target_y, target_w, target_h) = cv2.boundingRect(target_convex_hull)

                    # replace triangles from original to target
                    for i in indices_triangles:
                        # original
                        tr1_pt1 = landmarks_points[i[0]]
                        tr1_pt2 = landmarks_points[i[1]]
                        tr1_pt3 = landmarks_points[i[2]]
                        tr1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
                        (x, y, w, h) = cv2.boundingRect(tr1)

                        cropped_tr1 = img[y: y + h, x: x + w]
                        cropped_tr1_gray = cv2.cvtColor(cropped_tr1, cv2.COLOR_BGR2GRAY)
                        cropped_tr1_mask = np.zeros_like(cropped_tr1_gray)
                        points1 = [[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]]
                        cv2.fillConvexPoly(cropped_tr1_mask, np.array(points1, np.int32), (255, 255, 255))
                        cropped_tr1 = cv2.bitwise_and(cropped_tr1, cropped_tr1, mask=cropped_tr1_mask)

                        # target
                        tr2_pt1 = target_landmarks_points[i[0]]
                        tr2_pt2 = target_landmarks_points[i[1]]
                        tr2_pt3 = target_landmarks_points[i[2]]
                        tr2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
                        (x, y, w, h) = cv2.boundingRect(tr2)

                        cropped_tr2 = target[y: y + h, x: x + w]
                        cropped_tr2_mask = np.zeros((h, w), np.int8)
                        points2 = [[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                   [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                   [tr2_pt3[0] - x, tr2_pt3[1] - y]]
                        cv2.fillConvexPoly(cropped_tr2_mask, np.array(points2, np.int32), (255, 255, 255))
                        cropped_tr2 = cv2.bitwise_and(cropped_tr2, cropped_tr2, mask=cropped_tr2_mask)

                        # warp triangles (affine)
                        points1 = np.float32(points1)
                        points2 = np.float32(points2)
                        M = cv2.getAffineTransform(points1, points2)
                        warped_tr = cv2.warpAffine(cropped_tr1, M, (w, h))

                        # reconstruct final face
                        triangle_area = final[y: y + h, x: x + w]
                        for i in range(h):
                            for j in range(w):
                                for c in range(3):
                                    final[i + y, j + x, c] = max(final[i + y, j + x, c], warped_tr[i, j, c])

                    mask_target = np.zeros_like(target)
                    cv2.fillConvexPoly(mask_target, target_convex_hull, (255, 255, 255))
                    p = (target_x + target_w // 2, target_y + target_h // 2)
                    final = cv2.seamlessClone(final, target, mask_target, p, 1)
                cv2.imshow('Frame', final)
            else:
                cv2.imshow('Frame', target)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(30)

            if key == ord('q'):
                break
        else:
            break

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()
