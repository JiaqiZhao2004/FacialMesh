import cv2
import numpy as np
import dlib
from data import orientation_token_set, indices_triangles_set
from utils import extract_index

vid_capture = cv2.VideoCapture("roy.mov")
detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
grid = 10


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

initialization_data = dict()
for i in range(grid ** 2):
    initialization_data[i] = None
grid_completed = []

while vid_capture.isOpened():
    ret, img = vid_capture.read()
    if ret:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_faces = detector(img_gray)
        if len(img_faces) > 0:
            # print("face detected")
            for face in img_faces:
                landmarks = predictor(img_gray, face)
                img_landmarks_points = []
                max_x = 0
                min_x = 1e9
                max_y = 0
                min_y = 1e9
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    img_landmarks_points.append((x, y))
                    cv2.circle(img, (x, y), 2, (100, 100, 200), -1)
                    if x < min_x:
                        min_x = x
                    elif x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    elif y > max_y:
                        max_y = y
                cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 255), 1)
                # corners = str(face)[1:-1].replace("(", "").replace(")", "").replace(",", "").split(" ")
                # landmarks_points = [(int(corners[0]), int(corners[1])), (int(corners[0]), int(corners[3])), (int(corners[2]), int(corners[1])),
                #                     (int(corners[2]), int(corners[3]))]
                points = np.array(img_landmarks_points, np.int32)

                # log into grid
                grid_current = (round((points[30][0] - min_x) / (max_x - min_x) * grid), round((points[30][1] - min_y) / (max_y - min_y) * grid))
                grid_current_number = grid_current[0] + grid_current[1] * grid
                cv2.circle(img, (int((points[30][0] - min_x) / (max_x - min_x) * 200), int((points[30][1] - min_y) / (max_y - min_y) * 200)), 3, (0, 255, 0), -1)

                if initialization_data[grid_current_number] is None:
                    # orientation token
                    token = np.concatenate([points[:, 0] - points[30, 0], points[:, 1] - points[30, 1]])
                    token = (token - np.average(token)) / np.std(token)

                    # Delaunay Triangulation
                    convex_hull = cv2.convexHull(points)
                    rect = cv2.boundingRect(convex_hull)
                    sub_div = cv2.Subdiv2D(rect)
                    sub_div.insert(img_landmarks_points)
                    triangles = sub_div.getTriangleList()
                    triangles = np.array(triangles, np.int32)

                    # indices of triangles
                    indices_triangles = []
                    for t in triangles:
                        point1 = (t[0], t[1])
                        point2 = (t[2], t[3])
                        point3 = (t[4], t[5])
                        cv2.line(img, point1, point2, (0, 0, 255), 1)
                        cv2.line(img, point3, point2, (0, 0, 255), 1)
                        cv2.line(img, point1, point3, (0, 0, 255), 1)

                        index_point1 = extract_index(np.where((points == point1).all(axis=1)))
                        index_point2 = extract_index(np.where((points == point2).all(axis=1)))
                        index_point3 = extract_index(np.where((points == point3).all(axis=1)))

                        if index_point1 is not None and index_point2 is not None and index_point3 is not None:
                            indices_triangles.append((index_point1, index_point2, index_point3, point1, point2, point3))

                    cv2.putText(img, "Recorded", (200, 200), 1, 10, (0, 0, 0))
                    grid_completed.append(grid_current)
                    initialization_data[grid_current_number] = (token, indices_triangles)

                # print(token)
                # print("chin center = {}, {}".format(token[8], token[8 + 68]))
                # print("left eye = {}".format(token[38] - token[40]))
                # print("right eye = {}".format(token[44] - token[46]))
                # print("left sun = {}, {}".format(token[0], token[68))
                # print("right sun = {}".format(token[16], token[68 + 16]))
                # print("mouth = {}".format(token[62] - token[66]))

                # a.append(token[8])
                # b.append(token[8 + 68])
                # c.append(token[38] - token[40])
                # d.append(token[44] - token[46])
                # e.append(token[0])
                # f.append(token[68])
                # g.append(token[16])
                # h.append(token[16 + 68])
                # i.append(token[62] - token[66])
                #
                #
                # cv2.putText(img, "left eye = {}".format(round(token[38 + 68] - token[40 + 68], 3)), (100, 200), 2, 3, (255, 255, 0), 3)
                # cv2.putText(img, "right eye = {}".format(round(token[44 + 68] - token[46 + 68], 3)), (100, 300), 2, 3, (255, 255, 0), 3)
                # cv2.putText(img, "left sun = {}".format(round(token[0], 3)), (100, 400), 2, 3, (255, 255, 0), 3)
                # cv2.putText(img, "right sun = {}".format(round(token[16], 3)), (100, 500), 2, 3, (255, 255, 0), 3)
                # cv2.putText(img, "chin center = ({}, {})".format((round(token[8], 3)), round(token[8 + 68], 3)), (100, 600), 2, 3, (255, 255, 0), 3)
                # cv2.putText(img, "mouth = ({}, {})".format(round(token[62] - token[66], 3), round(token[62 + 68] - token[66 + 68], 3)), (100, 700), 2, 3, (255, 255, 0), 3)

                # compare with init tokens, find the most similar one
                # dist = np.array([np.linalg.norm(token - target) for target in orientation_token_set], np.float32)
                # target_set = indices_triangles_set[dist.argmin()]
                # x0 = 300
                # y0 = -500
                # for t_set in target_set:
                #     x1, y1 = t_set[3]
                #     x2, y2 = t_set[4]
                #     x3, y3 = t_set[5]
                    # cv2.circle(img, (x1 + x0, y1 + y0), 5, (255, 0, 0), -1)
                    # cv2.circle(img, (x2 + x0, y2 + y0), 5, (255, 0, 0), -1)
                    # cv2.circle(img, (x3 + x0, y3 + y0), 5, (255, 0, 0), -1)

        cv2.rectangle(img, (0, 0), (200, 200), (0, 0, 0), 1)
        for (x, y) in grid_completed:
            cv2.rectangle(img, (int((x - 1) * 200 / grid), int((y - 1) * 200 / grid)), (int(x * 200 / grid), int(y * 200 / grid)), (255, 0, 0), -1)
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

# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
# print(g)
# print(h)
# print(i)
