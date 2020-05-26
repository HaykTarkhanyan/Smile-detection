import os
import cv2
import re

ims = os.listdir(
    "/content/drive/My Drive/Smile_dataset/smile/lfwcrop_grey/faces")

with open("/content/drive/My Drive/Smile_dataset/smile/SMILE_list.txt") as names:
    smileys = names.read().splitlines()

with open("/content/drive/My Drive/Smile_dataset/smile/NON-SMILE_list.txt") as names:
    not_smileys = names.read().splitlines()


for i in ims:
    print(ims.index(i))
    if i in smileys and i in not_smileys:
        bugs += 0
    elif i in smileys:
        im = cv2.imread(
            f"/content/drive/My Drive/Smile_dataset/smile/lfwcrop_grey/faces/{i}")
        i = re.sub('.pgm$', '.png', i)
        cv2.imwrite(
            f'/content/drive/My Drive/Smile_dataset/smile/lfwcrop_grey/smile/{i}', im)
        smiley_names.append(i)
    elif i in not_smileys:
        im = cv2.imread(
            f"/content/drive/My Drive/Smile_dataset/smile/lfwcrop_grey/faces/{i}")
        i = re.sub('.pgm$', '.png', i)
        cv2.imwrite(
            f'/content/drive/My Drive/Smile_dataset/smile/lfwcrop_grey/not_smile/{i}', im)
        not_smiley_names.append(i)
