import os
import sys

import rawpy

import cv2
import time


def split_image(img, side=3):
    images = []
    for i in range(side):
        for j in range(side):
            temp = img[1024*i:1024*(i+1),1024*j:1024*(j+1),:]
            images.append(temp)
    return images


if __name__ == '__main__':
    src_name = 'asphalt-smoking-lot'
    target_name = src_name + '-split'

    src_folder = os.path.join('/home/sangwon/Downloads', src_name)
    target_folder = os.path.join('/home/sangwon/Downloads', target_name)

    try:
        os.mkdir(target_folder)
    except:
        pass

    files = [os.path.join(src_folder, f) for f in os.listdir(src_folder)]

    counter = 0
    for file in files:
        if file.endswith('.dng'):
            img = rawpy.imread(file)
            img = img.postprocess()
        else:
            img = cv2.imread(file)

        # Consider read photo size
        # Case 3024 x 3024
        if img.shape[0] == img.shape[1] == 3024:
            reshaped = cv2.resize(img, (3072, 3072))
            side = 3
        elif (img.shape[0] == 5760 and img.shape[1] == 4312) or (img.shape[0] == 4312 and img.shape[1] == 5760):
            reshaped = img[832:832 + 4096,108:108+4096,:]
            reshaped = cv2.resize(reshaped, (3072, 3072))
            side = 3
        elif (img.shape[0] == 4032 and img.shape[1] == 3024) or (img.shape[0] == 4032 and img.shape[1] == 3024):
            reshaped = img[504:504+3024,:,:]
            reshaped = cv2.resize(reshaped, (3072, 3072))
            side = 3

        splitted = split_image(reshaped, side=side)
        individual_count = 0
        for i in splitted:
            for rot in ['', cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                new_file_name = os.path.join(target_folder, str(time.time()) + '.png')

                if rot != '':
                    rot_i = cv2.rotate(i, rot)
                else:
                    rot_i = i

                cv2.imwrite(new_file_name, rot_i)
                individual_count += 1

        counter += 1
        print(counter / len(files))
