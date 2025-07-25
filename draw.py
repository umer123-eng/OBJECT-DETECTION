# Optical Flow Motion Detection Using calcOpticalFlowPyrLK
# Rearranged and cleaned version of the provided code

import cv2
import numpy as np
import argparse
import datetime

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(35, 35),
    maxLevel=7,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Parameters for ShiTomasi corner detection
feature_params = dict(
    maxCorners=1000,
    qualityLevel=0.6,
    minDistance=100,
    blockSize=7
)

class MotionApp:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0

        self.cam_no = int(input("Enter 1 for DAY, 2 for NOON, and 3 for NIGHT camera: "))

        video_paths = {
            1: "ambulance.mkv",
            2: "02-Mar-17 9_00_00 AM (UTC+05_30).mkv"
          }

        self.cam = cv2.VideoCapture(video_paths.get(self.cam_no, 0))

        ret, rstframe = self.cam.read()
        self.gray = cv2.cvtColor(rstframe, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("pexels-sohelpatel-1804035.jpg", rstframe)

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_gray, self.gray)

            if self.cam_no == 1:
                frame_diff[0:183, 0:230] = 0
                frame_diff[0:350, 400:640] = 0
                frame_diff[0:164, 0:640] = 0
                threshold_val = 30
                dilate_iter = 3
            elif self.cam_no == 2:
                frame_diff[0:270, 0:300] = 0
                frame_diff[0:350, 470:640] = 0
                frame_diff[0:215, 0:640] = 0
                threshold_val = 30
                dilate_iter = 3
            elif self.cam_no == 3:
                frame_diff[0:375, 0:110] = 0
                frame_diff[0:150, 0:400] = 0
                frame_diff[0:375, 370:800] = 0
                threshold_val = 25
                dilate_iter = 2

            _, im_thresh = cv2.threshold(frame_diff, threshold_val, 255, cv2.THRESH_BINARY)
            im_thresh = cv2.dilate(im_thresh, None, iterations=dilate_iter)
            im_thresh = cv2.medianBlur(im_thresh, 9)

            cv2.imshow('Foreground Mask', im_thresh)

            contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 5:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            vis = frame.copy()
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.ones_like(im_thresh) * 255
                p = cv2.goodFeaturesToTrack(im_thresh, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            cv2.putText(vis, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.imshow('Motion Detection', vis)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    MotionApp().run()
