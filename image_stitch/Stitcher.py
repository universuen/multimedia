import cv2
import numpy as np
import os


class Stitcher:
    def __init__(self, path, ratio=0.5):
        # 读取所有图片
        self.algorithm = cv2.RANSAC
        self.ratio = ratio
        self.images = list()
        filenames = os.listdir(path)
        for i in filenames:
            file_path = path + "\\" + i
            img = cv2.imread(file_path)
            # img = cv2.resize(img, (480, 320))
            self.images.append(img)
        # self.images.reverse()

    def stitch(self):
        # 缝合所有图片
        result = self.images[0]
        for img in self.images[1:]:
            result = self.stitch_two(result, img)
        return result

    def stitch_two(self, img_a, img_b):
        keypoints_a, descriptors_a = self._analyze(img_a)
        keypoints_b, descriptors_b = self._analyze(img_b)
        homography = self._match(keypoints_a, descriptors_a, keypoints_b, descriptors_b)
        inverse_homography = np.linalg.inv(homography)
        f1 = np.dot(inverse_homography, np.array([0, 0, 1]))
        f1 /= f1[-1]
        inverse_homography[0][-1] += abs(f1[0])
        inverse_homography[1][-1] += abs(f1[1])
        dimension_size = np.dot(inverse_homography, np.array([img_a.shape[1], img_a.shape[0], 1]))
        offset_x = abs(int(f1[0]))
        offset_y = abs(int(f1[1]))
        dimension_size = (
        int(dimension_size[0]) + offset_x + img_b.shape[1], int(dimension_size[1]) + offset_y + img_b.shape[0])
        result = cv2.warpPerspective(img_a, inverse_homography, dimension_size)
        result[offset_y:offset_y + img_b.shape[0], offset_x:offset_x + img_b.shape[1]] = img_b
        result = self._move(result)
        return result

    def _analyze(self, image):  # -> keypoints, descriptors
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(gray_img, None)

    def _match(self, keypoints_b, descriptors_b, keypoints_a, descriptors_a):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=100)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))
        if len(good_matches) > 4:
            good_keypoints_a = np.float32([keypoints_a[i].pt for (_, i) in good_matches])
            good_keypoints_b = np.float32([keypoints_b[i].pt for (i, _) in good_matches])
            homography, _ = cv2.findHomography(good_keypoints_a, good_keypoints_b, self.algorithm)
            return homography
        else:
            return None

    def _move(self, image):
        offset = []
        for i in range(image.shape[0]):
            if not np.all(image[i] == 0):
                offset.append(i)
                break
        for i in range(image.shape[0] - 1, -1, -1):
            if not np.all(image[i] == 0):
                offset.append(i)
                break
        for i in range(image.shape[1]):
            if not np.all(image[i] == 0):
                offset.append(i)
                break
        for i in range(image.shape[1] - 1, -1, -1):
            if not np.all(image[:, i] == 0):
                offset.append(i)
                break
        image = image[offset[0]:offset[1], offset[2]:offset[3]]
        return image


if __name__ == '__main__':
    stitcher = Stitcher("./images")
    result = stitcher.stitch()
    cv2.imshow("Result", result)
    cv2.waitKey()
