from init import initialize
import cv2
import numpy as np
import dlib
from utils import *
from data import orientation_token_set, indices_triangles_set
TARGET = "Untitled.mov"
# TARGET = 0

# orientation_token_set, indices_triangles_set = initialize()

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(TARGET)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

print("2")

while vid_capture.isOpened():
    ret, img = vid_capture.read()
    if ret:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final = np.zeros_like(img)
        img_faces = detector(img_gray)
        if len(img_faces) > 0:
            print("face detected")
            for face in img_faces:
                landmarks = predictor(img_gray, face)
                img_landmarks_points = []
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    img_landmarks_points.append((x, y))
                print("3")

                points = np.array(img_landmarks_points, np.int32)
                # orientation token
                token = np.concatenate([points[:, 0] - points[30, 0], points[:, 1] - points[30, 1]])
                token = (token - np.average(token)) / np.std(token)

                # compare with init tokens, find the most similar one
                print("4")
                dist = np.array([np.linalg.norm(token - target) for target in orientation_token_set], np.float32)
                target_set = indices_triangles_set[dist.argmin()]
                print("5")
                x0 = 300
                y0 = -500
                for t_set in target_set:
                    x1, y1 = t_set[3]
                    x2, y2 = t_set[4]
                    x3, y3 = t_set[5]
                    cv2.circle(img, (x1 + x0, y1 + y0), 5, (255, 0, 0), -1)
                    cv2.circle(img, (x2 + x0, y2 + y0), 5, (255, 0, 0), -1)
                    cv2.circle(img, (x3 + x0, y3 + y0), 5, (255, 0, 0), -1)
                print("6")

                # # convex hull
                # target_points = np.array(img_landmarks_points, np.int32)
                # target_convex_hull = cv2.convexHull(target_points)
                # (target_x, target_y, target_w, target_h) = cv2.boundingRect(target_convex_hull)
                #
                # # replace triangles from original to target
                # for i in indices_triangles:
                #     # original
                #     tr1_pt1 = landmarks_points[i[0]]
                #     tr1_pt2 = landmarks_points[i[1]]
                #     tr1_pt3 = landmarks_points[i[2]]
                #     tr1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
                #     (x, y, w, h) = cv2.boundingRect(tr1)
                #
                #     cropped_tr1 = img[y: y + h, x: x + w]
                #     cropped_tr1_gray = cv2.cvtColor(cropped_tr1, cv2.COLOR_BGR2GRAY)
                #     cropped_tr1_mask = np.zeros_like(cropped_tr1_gray)
                #     points1 = [[tr1_pt1[0] - x, tr1_pt1[1] - y],
                #                [tr1_pt2[0] - x, tr1_pt2[1] - y],
                #                [tr1_pt3[0] - x, tr1_pt3[1] - y]]
                #     cv2.fillConvexPoly(cropped_tr1_mask, np.array(points1, np.int32), (255, 255, 255))
                #     cropped_tr1 = cv2.bitwise_and(cropped_tr1, cropped_tr1, mask=cropped_tr1_mask)
                #
                #     # target
                #     tr2_pt1 = target_landmarks_points[i[0]]
                #     tr2_pt2 = target_landmarks_points[i[1]]
                #     tr2_pt3 = target_landmarks_points[i[2]]
                #     tr2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
                #     (x, y, w, h) = cv2.boundingRect(tr2)
                #
                #     cropped_tr2 = target[y: y + h, x: x + w]
                #     cropped_tr2_mask = np.zeros((h, w), np.int8)
                #     points2 = [[tr2_pt1[0] - x, tr2_pt1[1] - y],
                #                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                #                [tr2_pt3[0] - x, tr2_pt3[1] - y]]
                #     cv2.fillConvexPoly(cropped_tr2_mask, np.array(points2, np.int32), (255, 255, 255))
                #     cropped_tr2 = cv2.bitwise_and(cropped_tr2, cropped_tr2, mask=cropped_tr2_mask)
                #
                #     # warp triangles (affine)
                #     points1 = np.float32(points1)
                #     points2 = np.float32(points2)
                #     M = cv2.getAffineTransform(points1, points2)
                #     warped_tr = cv2.warpAffine(cropped_tr1, M, (w, h))
                #
                #     # reconstruct final face
                #     triangle_area = final[y: y + h, x: x + w]
                #     for i in range(h):
                #         for j in range(w):
                #             for c in range(3):
                #                 final[i + y, j + x, c] = max(final[i + y, j + x, c], warped_tr[i, j, c])
                #
                # mask_target = np.zeros_like(target)
                # cv2.fillConvexPoly(mask_target, target_convex_hull, (255, 255, 255))
                # p = (target_x + target_w // 2, target_y + target_h // 2)
                # final = cv2.seamlessClone(final, target, mask_target, p, 1)

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
