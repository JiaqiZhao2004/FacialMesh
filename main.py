import cv2
import numpy as np
import dlib
from data import orientation_token_set, indices_triangles_set

vid_capture = cv2.VideoCapture(0)
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

while vid_capture.isOpened():
    ret, img = vid_capture.read()
    if ret:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

                points = np.array(img_landmarks_points, np.int32)
                # orientation token
                token = np.concatenate([points[:, 0] - points[30, 0], points[:, 1] - points[30, 1]])
                token = (token - np.average(token)) / np.std(token)

                print("nose tip = {}".format(token[30]))
                print("left eye = {}".format(token[38] - token[40]))
                print("right eye = {}".format(token[44] - token[46]))
                print("left sun = {}".format(token[0]))
                print("right sun = {}".format(token[16]))
                print("chin center = {}".format(token[8]))
                print("mouth = {}".format(token[62] - token[66]))

                cv2.putText(img, "nose tip = {}".format(token[30]), (100, 100), 2, 3, (255, 255, 0), 3)
                cv2.putText(img, "left eye = {}".format(token[38] - token[40]), (100, 200), 2, 3, (255, 255, 0), 3)
                cv2.putText(img, "right eye = {}".format(token[44] - token[46]), (100, 300), 2, 3, (255, 255, 0), 3)
                cv2.putText(img, "left sun = {}".format(token[0]), (100, 400), 2, 3, (255, 255, 0), 3)
                cv2.putText(img, "right sun = {}".format(token[16]), (100, 500), 2, 3, (255, 255, 0), 3)
                cv2.putText(img, "chin center = {}".format(token[8]), (100, 600), 2, 3, (255, 255, 0), 3)
                cv2.putText(img, "mouth = {}".format(token[62] - token[66]), (100, 700), 2, 3, (255, 255, 0), 3)

                # compare with init tokens, find the most similar one
                dist = np.array([np.linalg.norm(token - target) for target in orientation_token_set], np.float32)
                target_set = indices_triangles_set[dist.argmin()]
                x0 = 300
                y0 = -500
                for t_set in target_set:
                    x1, y1 = t_set[3]
                    x2, y2 = t_set[4]
                    x3, y3 = t_set[5]
                    cv2.circle(img, (x1 + x0, y1 + y0), 5, (255, 0, 0), -1)
                    cv2.circle(img, (x2 + x0, y2 + y0), 5, (255, 0, 0), -1)
                    cv2.circle(img, (x3 + x0, y3 + y0), 5, (255, 0, 0), -1)

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
