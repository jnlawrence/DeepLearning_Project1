"""
SAM & Clip Client

Usage: 
    python3 client --image <path>

Author:
    Rowel Atienza rowel@eee.upd.edu.ph

    John  Lawrence Abarquez jbabarquez1@up.edu.ph
"""
import argparse
import logging
import cv2
import numpy as np
import urllib.request
import validators
import numpy as np
from pycocotools import mask as maskUtils
import pycocotools._mask as _mask
import os

from pytriton.client import ModelClient

logger = logging.getLogger("SAM Client")

def sam_model(args):
    with ModelClient(args.url, args.model, init_timeout_s=args.init_timeout_s) as client:
        if validators.url(args.image):
            with urllib.request.urlopen(args.image) as url_response:
                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, -1)
        else:
            image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.expand_dims(image, axis=0)
        logger.info(f"Running inference requests")
        masks = client.infer_sample(image)
        segmentation = masks['segmentation']
        area = masks['area']

        for k, v in masks.items():
            print(k, v.shape)
        
        def toBbox(rleObjs):
            if type(rleObjs) == list:
                return _mask.toBbox(rleObjs)
            else:
                return _mask.toBbox([rleObjs])[0]
        
        def encode(bimask):
            if len(bimask.shape) == 3:
                return _mask.encode(bimask)
            elif len(bimask.shape) == 2:
                h, w = bimask.shape
                return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]
            
        
        dict = {}
        bbox_arr = []
        area_arr = []
        dict_list=[]
        img = cv2.imread("images/dog_car.jpg")
        with open('bbox.txt', 'w') as f:
            for i in range(len(segmentation)):
                #add area of mask to area array
                area_arr.append(area[i])
                #convert to encodable format
                fort = np.asfortranarray(segmentation[i])  
                sample1 = encode(fort)
                #from RLE convert to bbox
                bbox = toBbox(sample1)
                # print(bbox)
                #Add bbox to bbox array
                bbox_arr.append(bbox)

                #write bbox to file
                # Replace filename with 'bbox' at final output
                dict = {str(i) + ".bmp": bbox}

                dict_list.append(dict)

                # f.write(str(dict) + ",")

                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                # print(x, y, w, h)
                crop_img = img[y:y+h, x:x+w]
                # print(crop_img)
                # cv2.imshow("cropped", crop_img[i])
                # path = 'DeepLearning_Project1/mlops/triton/openclip/crop'
                # if not cv2.imwrite(os.path.join('DeepLearning_Project1/mlops/triton/openclip/crop', str(i) + ".bmp"), crop_img):
                #     raise Exception("Could not write image")
            # print(bbox_arr)
        print(dict_list)


logger = logging.getLogger("OpenClip & CoCa")

label_path = "/mnt/c/Windows/System32/repos/DeepLearning_Project1/mlops/triton/openclip/imagenet2coco.txt"
def clip_model(args):
    with ModelClient(args.url, args.openclip_model, init_timeout_s=args.init_timeout_s) as client:
        if validators.url(args.image):
            with urllib.request.urlopen(args.image) as url_response:
                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, -1)
        else:
            image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.expand_dims(image, axis=0)
        logger.info(f"Running inference requests")
        outputs = client.infer_sample(image)
        for k, v in outputs.items():
            if k == "index":
                print(k, v, v.shape)
            else:
                #Label of picture
                label = v.tobytes().decode('utf-32')
                print(label)

                #Convert ImageNet label to coco label
                def search_str(file_path, word):
                    with open(file_path, 'r') as file:
                        for num, line in enumerate(file, 1):
                            # check if string present in a file
                            if word in line:
                                # print('word exist in line ' + str(num))
                                sentence = line.split()
                                new_label = sentence[-1]
                                print("inside funct new label is " + new_label)
                                return new_label

                x = search_str(label_path, label)
                print("outside new label is " + x)

                
    with ModelClient(args.url, args.coca_model, init_timeout_s=args.init_timeout_s) as client:
        if validators.url(args.image):
            with urllib.request.urlopen(args.image) as url_response:
                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, -1)
        else:
            image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Running inference requests")
        outputs = client.infer_sample(image)
        for k, v in outputs.items():
            print(v.tobytes().decode('utf-32'))

    
                

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="crop/86.bmp",
        help=(
            "Path to image can filesystem path or url path"
        ),
    )
    choices = ["OpenClip_b32"]
    parser.add_argument(
        "--openclip-model",
        default=choices[0],
        choices=choices,
        help=(
            "OpenClip model" 
        ),
    )
    choices = ["CoCa_l14"]
    parser.add_argument(
        "--coca-model",
        default=choices[0],
        choices=choices,
        help=(
            "CoCa model" 
        ),
    )
    parser.add_argument(
        "--url",
        default="http://202.92.132.48:8000",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent requests.",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    sam_model(args)
    clip_model(args)

if __name__ == "__main__":
    main()
