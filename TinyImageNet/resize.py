import cv2
import glob

def resize_img(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    image_path_1 = image_path.split('/')[-1]
    image_path_2 = image_path.split('/')[0:-2]
    image_path_2 = '/'.join(image_path_2)
    image_path = image_path_2 + '/' + image_path_1
    print(image_path)
    cv2.imwrite(image_path,img)
    
all_images = glob.glob('/content/tiny-imagenet-200/*/*/*/*')
for image in all_images:
    resize_img(image, 32)