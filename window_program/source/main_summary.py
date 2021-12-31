
# coding: utf-8

import sys 
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


from utils_inference import get_lmks_by_img, get_model_by_name, get_preds, decode_preds, crop, detect_direction
from utils_landmarks import show_landmarks, get_five_landmarks_from_net, alignment_orig, get_six_landmarks_from_net

import os
from os.path import join as ospj

def Area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def calc_area(img):
    lmks = img['lmks']
    left_eye  = lmks[0]
    right_eye = lmks[1]
    nose      = lmks[2]
    left_lip  = lmks[3]
    right_lip = lmks[4]
    
    return Area([nose, left_eye, right_eye]), Area([nose, left_lip, right_lip])

def get_lmks_with_dicts(model, input_path, img_paths):
    def calc_dist(start,end):
        return ((end[0]-start[0])**2 + (end[1]-start[1])**2)**(1/2)
    
    def check_nose_right(lmks):
        left_eye  = lmks[0]
        right_eye = lmks[1]
        nose      = lmks[2]
        left_lip  = lmks[3]
        right_lip = lmks[4]
        
        eye = (left_eye + right_eye)/2
        lip = (left_lip + right_lip)/2
        center = (left_eye + right_eye + left_lip + right_lip)/4
        
        if calc_dist(center, nose) > calc_dist(center, eye):
            lmks[2] = center
        
        return lmks

    i = 0
    imgs_dict={}
    for img_path in img_paths:
        i +=1
        print(ospj(input_path,img_path))
        img_dict = {}
        img = cv2.imread(ospj(input_path,img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_dict['img'] = img
        img_dict['width'] = img.shape[1]
        img_dict['height'] = img.shape[0]

        lmks = get_lmks_by_img(model, img) 
        five_lmks = get_six_landmarks_from_net(lmks)
        lmks = check_nose_right(five_lmks)
        img_dict['lmks'] = lmks

        img_dict['top_area'], img_dict['bottom_area'] = calc_area(img_dict)

        left_eye  = lmks[0]
        right_eye = lmks[1]
        left_lip  = lmks[3]
        right_lip = lmks[4]

        img_dict["direction"], _ = detect_direction(lmks)
        img_dict["eye2eye"] = calc_dist(left_eye, right_eye)
        img_dict["eye2lip"] = calc_dist((left_eye+right_eye)/2, (left_lip+right_lip)/2)
        img_dict["lip2lip"] = calc_dist(left_lip, right_lip)
        imgs_dict[img_path] = img_dict
    return imgs_dict

def sort_imgs_to_set(imgs_dict):
    landscape_d = {}
    zoom_in_d ={}
    zoom_out_d = {}
    # landscape
    # ready to split btw zoom_in & zoom_out
    distance_btw_lips = []
    distance_btw_eye2lip = []
    for k in imgs_dict.keys():
        width = imgs_dict[k]['width']
        height = imgs_dict[k]['height']
        lmks = imgs_dict[k]['lmks']

        if width > height:
            landscape_d[k] = imgs_dict[k]
        else:
            left_eye = lmks[0]
            right_eye = lmks[1]
            left_lip = lmks[3]
            right_lip = lmks[4]
            distance_btw_lips.append(imgs_dict[k]["lip2lip"])
            distance_btw_eye2lip.append(imgs_dict[k]["eye2lip"])
   
    import matplotlib.pyplot as plt
    bound_w = plt.hist(distance_btw_lips, bins=2)
    bound_h = plt.hist(distance_btw_eye2lip, bins=2)
    half_w = bound_w[1][1]
    half_h = bound_h[1][1]
    
    #zoom_in / zoom_out
    for k in imgs_dict.keys():
        width = imgs_dict[k]['width']
        height = imgs_dict[k]['height']
        lmks = imgs_dict[k]['lmks']

        left_eye  = lmks[0]
        right_eye = lmks[1]
        left_lip  = lmks[3]
        right_lip = lmks[4]

        if width < height:
            btw_lips = abs(left_lip[0] - right_lip[0])
            btw_eye2lip = abs((left_lip[1]+right_lip[1])/2 - (left_eye[1]+right_eye[1])/2)
            if btw_lips > half_w or btw_eye2lip > half_h:
                zoom_in_d[k] = imgs_dict[k]
            else:
                zoom_out_d[k] = imgs_dict[k]
    return landscape_d, zoom_in_d, zoom_out_d

def sort_set_to_dirtype(target_d):
    def calc_dist(start,end):
        return ((end[0]-start[0])**2 + (end[1]-start[1])**2)**(1/2)
    print(target_d.keys())
    h_d = {}
    v_d = {}
    o_d = {}
    _temp_l = []
    
    for k in target_d.keys():
        target = target_d[k]
        direction = target["direction"]
        if direction != 2:
            h_d[k] = target_d[k]
        else:  
            lmks = target_d[k]['lmks']
            left_eye  = lmks[0]
            right_eye = lmks[1]
            nose      = lmks[2]
            left_lip  = lmks[3]
            right_lip = lmks[4]
            updown_ratio = target_d[k]["top_area"]
            _temp_l.append(updown_ratio)

    boundary = plt.hist(_temp_l, bins=2)[1][1]

    for k in target_d.keys():
        target = target_d[k]
        direction = target["direction"]
        if direction == 2:
            lmks = target_d[k]['lmks']
            left_eye  = lmks[0]
            right_eye = lmks[1]
            nose      = lmks[2]
            left_lip  = lmks[3]
            right_lip = lmks[4]
            updown_ratio = target_d[k]["top_area"]
            if updown_ratio > boundary :
                o_d[k] = target_d[k]
            else:
                v_d[k] = target_d[k]
   
    target_split_d = {}          
    target_split_d["origin"] = o_d
    target_split_d["horizontal"] = h_d
    target_split_d["vertical"] = v_d
    return target_split_d


def sort_all_sets_to_dirtype(landscape_d, zoom_in_d, zoom_out_d):
    landscape = sort_set_to_dirtype(landscape_d)
    zoom_in   = sort_set_to_dirtype(zoom_in_d)
    zoom_out  = sort_set_to_dirtype(zoom_out_d)
    return landscape, zoom_in, zoom_out

def set_scale_ref_img(target_dicts):
    def find_max_length(target_dicts):
        max_length = 0
        ref_img = 0
        for k in target_dicts["origin"].keys():
            target = target_dicts["origin"][k]
            length = target["eye2lip"]
            if max_length < length:
                max_length = length
                ref_img = k
        return ref_img
    
    def find_min_length(target_dicts):
        min_length = 100000000
        ref_img = 0
        for k in target_dicts["origin"].keys():
            target = target_dicts["origin"][k]
            length = target["eye2lip"]
            if min_length > length:
                min_length = length
                ref_img = k
        return ref_img
    
    
    ref_key = find_max_length(target_dicts)
    target_dicts["ref_scale_ori"] = ref_key
    print("ref_key %s" % ref_key)
    for dir_type in ["origin", "horizontal", "vertical"]:
        if len(target_dicts[dir_type].keys()) == 0:
            continue
        for k in target_dicts[dir_type].keys():
            target = target_dicts[dir_type][k]
            lmks = target["lmks"]
            width = target["width"]
            height = target["height"]
            
            if dir_type == "vertical":
                ref_where = "lip2lip"
            else:
                ref_where = "eye2lip"
                
            scale = target_dicts["origin"][ref_key][ref_where]/target[ref_where]
            #scale = 1.0
            scaled_img = cv2.resize(target["img"], dsize=(int(width*scale), int(height*scale)), interpolation=cv2.INTER_CUBIC)
            target["scale"] = scale
            target["crop_img"] = scaled_img


def shift_to_boardcenter(target_dict):
    board_w, board_h = None, None
    for k in target_dict.keys():
        target = target_dict[k]
        
        if board_w is None:
            board_w = target["width"]*3//2
            board_h = target["height"]*3//2
            
        shift_w = (board_w - target["width"])//2
        shift_h = (board_h - target["height"])//2
        
        T = [ [1, 0, shift_w],
             [0, 1, shift_h] 
            ]
        T = np.float32(T)
        transformed_img = cv2.warpAffine(target["img"], T, (board_w, board_h))
        target["crop_img"] = transformed_img
    return board_w, board_h

def shift_to_boardcenter_all(target_dicts):
    board_h, board_w = shift_to_boardcenter(target_dicts["origin"])
    board_h, board_w = shift_to_boardcenter(target_dicts["horizontal"])
    board_h, board_w = shift_to_boardcenter(target_dicts["vertical"])
    return board_h, board_w

def set_ref_points(lmks, align_where):
    eye = (lmks[0] + lmks[1])/2
    nose = lmks[2]
    lip = (lmks[3] + lmks[4])/2
    
  
    if align_where == "eye":
        '''
        ratio_a, ratio_b = 1, 2
        return eye/(ratio_a+ratio_b)*(ratio_b) + nose/(ratio_a+ratio_b)*(ratio_a)
        '''
        ratio_a, ratio_b = 1, 4
        return eye/(ratio_a+ratio_b)*(ratio_b) + lip/(ratio_a+ratio_b)*(ratio_a)
    
    elif align_where == "nose":
        ratio_a, ratio_b = 2, -1
        return eye/(ratio_a+ratio_b)*(ratio_b) + nose/(ratio_a+ratio_b)*(ratio_a)
    '''
    elif align_where == "chin":
        ratio_a, ratio_b = 1, 1
        return eye/(ratio_a+ratio_b)*(ratio_b) + chin/(ratio_a+ratio_b)*(ratio_a)
    '''


def set_ref_img(target_dicts, align_where):
    max_shift_w = 0
    max_shift_h = 0
    ref_img = 0
    
    dir_type = "origin"
    for k in target_dicts[dir_type].keys():
        target = target_dicts[dir_type][k]
        scale = target["scale"]
        lmks = scale * target_dicts[dir_type][k]["lmks"]
        eye = (lmks[0] + lmks[1])/2
        nose = lmks[2]
        lip = (lmks[3] + lmks[4])/2

        ref_points = set_ref_points(lmks, align_where)
        shift_w = scale * target['width']/2 - ref_points[0]
        shift_h = scale * target['height']/2 - ref_points[1]

        if abs(max_shift_w) < abs(shift_w):
            max_shift_w = shift_w
            ref_img = k
        if abs(max_shift_h) < abs(shift_h):
            max_shift_h = shift_h
            ref_img = k

    target_dicts["shift_w"] = max_shift_w
    target_dicts["shift_h"] = max_shift_h
    target_dicts["ref_ori"] = ref_img


def align_each(target_dicts, dir_type, align_where):
    if len(target_dicts[dir_type].keys()) == 0:
        return False
    board_w = target_dicts["board_w"]
    board_h = target_dicts["board_h"]
    ref_ori = target_dicts["ref_ori"]
    ref_scale = target_dicts["origin"][ref_ori]["scale"]
    ref_lmks  = ref_scale * target_dicts["origin"][ref_ori]["lmks"]
    ref_width = ref_scale * target_dicts["origin"][ref_ori]["width"]
    ref_height = ref_scale * target_dicts["origin"][ref_ori]["height"]
    
    ref_shift_w = target_dicts["shift_w"]
    ref_shift_h = target_dicts["shift_h"]
    
    crop_shift_w = 0
    crop_shift_h = 0
    
    ref_key = 0
    for k in target_dicts[dir_type].keys():
        target = target_dicts[dir_type][k]
        
        scale = target["scale"]
        width = scale * target["width"]
        height = scale * target["height"]
        lmks = scale * target["lmks"]
        eye = (lmks[0] + lmks[1])/2
        nose = lmks[2]
        lip = (lmks[3] + lmks[4])/2
        #chin = lmks[5]
        
        if align_where == "eye":
            ref_where = (ref_lmks[0] + ref_lmks[1])/2
            _align_where = eye
            '''
            if chin[0] is None:
                ref_where = (ref_lmks[0] + ref_lmks[1])/2
                _align_where = eye
            else:
                ref_where = ref_lmks[5]
                _align_where = chin
            ''' 
        elif align_where == "nose":
            ref_where = ref_lmks[2]
            _align_where = nose
        elif align_where == "lip":
            ref_where = (ref_lmks[3] + ref_lmks[4])/2
            _align_where = lip
        #elif align_where == "chin":
        #    ref_where = ref_lmks[5]
        #    _align_where = chin
        else:
            raise ValueError

        if dir_type == "origin":
            _align_where = set_ref_points(lmks, align_where)
            shift_w = width/2 - _align_where[0]
            shift_h = height/2 - _align_where[1]
        elif dir_type == "horizontal":
            shift_w = 0
            shift_h = ref_where[1] + ref_shift_h - _align_where[1] - (ref_height - height)/2
        elif dir_type == "vertical":
            shift_w = ref_where[0] + ref_shift_w - _align_where[0] - (ref_width - width)/2
            shift_h = 0 
        else:
            raise ValueError
        
        #update crop_shift
        if dir_type == "origin":
            if abs(shift_w) > abs(crop_shift_w):
                crop_shift_w = shift_w
                ref_key = k
            if abs(shift_h) > abs(crop_shift_h):
                crop_shift_h = shift_h
                ref_key = k
        elif dir_type == "horizontal":
            if abs(shift_h) > abs(crop_shift_h):
                crop_shift_h = shift_h
                ref_key = k
        elif dir_type == "vertical":
            if abs(shift_w) > abs(crop_shift_w):
                crop_shift_w = shift_w
                ref_key = k

        deg = 0
        if dir_type == "vertical" or dir_type == "origin":
            left_eye = lmks[0]
            right_eye = lmks[1]
            diff_eye2nose = abs(eye-nose)
            direction = np.argmax(diff_eye2nose)

            diff_eye = right_eye - left_eye
            if diff_eye[direction] != 0:
                if direction == 0:
                    deg = -1*diff_eye[1]/diff_eye[0]
                else:
                    deg = diff_eye[0]/diff_eye[1]
            deg = np.arctan(deg)
                
        R = cv2.getRotationMatrix2D((width/2,  height/2), deg, 1)
        T = [ [0, 0, shift_w],
             [0, 0, shift_h] 
            ]
        T = np.float32(T)
        
        
        transformed_img = cv2.warpAffine(target["crop_img"], R+T, (board_w, board_h))
        
        #_, updown_ratio = detect_direction(lmks)
        #print("dir_type {}, img {}, scale {}, updown_ratio {}".format(dir_type, k, target["scale"], updown_ratio))

        target["crop_img"] = transformed_img
    
    if "crop_shift_w" in target_dicts.keys():
        if abs(crop_shift_w) > abs(target_dicts["crop_shift_w"]):
            target_dicts["crop_shift_w"] = crop_shift_w
            target_dicts["ref_crop_img"] = ref_key
            target_dicts["ref_crop_dir"] = dir_type
            
        if abs(crop_shift_h) > abs(target_dicts["crop_shift_h"]):
            target_dicts["crop_shift_h"] = crop_shift_h
            target_dicts["ref_crop_img"] = ref_key
            target_dicts["ref_crop_dir"] = dir_type

    else:
        target_dicts["crop_shift_w"] = crop_shift_w
        target_dicts["crop_shift_h"] = crop_shift_h
        target_dicts["ref_crop_img"] = ref_key
        target_dicts["ref_crop_dir"] = dir_type
    
    return True
    

def align(target_dicts, align_where):
    align_each(target_dicts, "origin", align_where)
    align_each(target_dicts, "horizontal", align_where)
    align_each(target_dicts, "vertical", align_where)
    

def crop_each(target_dicts, dir_type):
    ref_crop_shift_w = target_dicts["crop_shift_w"] 
    ref_crop_shift_h = target_dicts["crop_shift_h"] 

    board_w = target_dicts["board_w"]
    board_h = target_dicts["board_h"]

    for k in target_dicts[dir_type].keys():
        target = target_dicts[dir_type][k]
        scale = target['scale']
        width = target['width']
        height = target['height']

        offset_w = width//2 - abs(ref_crop_shift_w)
        offset_h = height//2 - abs(ref_crop_shift_h)
        target["crop_img"] = target["crop_img"][int(board_h//2 - offset_h):int(board_h//2 + offset_h),
                                                int(board_w//2 - offset_w):int(board_w//2 + offset_w)]


def crop(target_dicts):
    crop_each(target_dicts, "origin")
    crop_each(target_dicts, "horizontal")
    crop_each(target_dicts, "vertical")



def edit_all_images(target_dicts, align_where):
    
    set_scale_ref_img(target_dicts)
    #"crop_img" generate
    board_w, board_h = shift_to_boardcenter_all(target_dicts)
    target_dicts["board_w"] = board_w
    target_dicts["board_h"] = board_h
    
    #"shift_w" ,"shift_h", "ref_ori" generate
    set_ref_img(target_dicts, align_where)
    
    # align : update @ "crop_img"
    align(target_dicts, align_where)
    crop(target_dicts)


def save_images(input_path, output_folder, target_dicts, dir_path="landscape"):
    
    _dir_path = ospj(input_path, output_folder)
    if not os.path.exists(_dir_path):
        os.mkdir(_dir_path)
    
    
    dir_path = ospj(_dir_path,dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    group_l = ["origin", "vertical", "horizontal"]
    for group in group_l:
        for k in target_dicts[group].keys():
            target = target_dicts[group][k]
            type = os.path.splitext(k)[1]

            result, encoded_img = cv2.imencode(type, cv2.cvtColor(target["crop_img"], cv2.COLOR_BGR2RGB))
            if result:
                with open(ospj(dir_path, str(k)), mode='w+b') as f:
                    encoded_img.tofile(f)

def save_log(path, output_path, set_list):
    import copy
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    dir, file = os.path.split(path)

    path = ospj(path, output_path)
    log_path = ospj(path,file+"_log.json")
    _landscape = copy.deepcopy(set_list[0])
    _zoom_in = copy.deepcopy(set_list[1])
    _zoom_out = copy.deepcopy(set_list[2])
    
    total = []
    total_cnt = 0
    for target_d in [_landscape, _zoom_in, _zoom_out]:
        cnt = {} 
        for dir_type in ["origin", "horizontal", "vertical"]:
            for target in target_d[dir_type].keys():
                del(target_d[dir_type][target]["img"])
                del(target_d[dir_type][target]["crop_img"])
            cnt[dir_type] = len(target_d[dir_type].keys())
            total_cnt += len(target_d[dir_type].keys())     
        total.append(cnt)

    all_dicts = {"path":path,
                "landscape":_landscape, 
                "zoom_in":_zoom_in,
                "zoom_out":_zoom_out
                }
    with open(log_path, 'w') as log_f :
        json.dump(json.dumps(all_dicts, cls=NumpyEncoder), log_f)

    print("total : {}\n".format(total_cnt))
    print("landscape : {} (O : {}, H : {}, V: {})\n".format((total[0]["origin"]+total[0]["horizontal"]+total[0]["vertical"]), total[0]["origin"], total[0]["horizontal"], total[0]["vertical"]))
    print("zoom_in : {} (O : {}, H : {}, V: {})\n".format((total[1]["origin"]+total[1]["horizontal"]+total[1]["vertical"]), total[1]["origin"], total[1]["horizontal"], total[1]["vertical"]))
    print("zoom_out : {} (O : {}, H : {}, V: {})\n".format((total[2]["origin"]+total[2]["horizontal"]+total[2]["vertical"]), total[2]["origin"], total[2]["horizontal"], total[2]["vertical"]))  



def main_job(input_path):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    input_path = input_path.replace("/", "\\")
    file_list = os.listdir(input_path)
    file_list = [file.lower() for file in file_list]
    img_paths = [file for file in file_list if file.lower().endswith(".jpg") or file.endswith(".jpeg")]
    
    if len(img_paths) == 0:
        print ("No Images!")
        return None

    #landscape_split_d/ zoom_in_split_d/ zoom_out_split_d
    model = get_model_by_name('AFLW')

    print ("="*10, " Loading Images... ", "="*10)
    imgs_dict = get_lmks_with_dicts(model, input_path, img_paths)

    #generate "landscape/ zoom_in/ zoom_out"
    print ("="*10, " Sort Images..... ", "="*10)
    landscape, zoom_in, zoom_out = sort_imgs_to_set(imgs_dict)

    #generate "origin/ horizontal/ vertical"
    landscape, zoom_in, zoom_out = sort_all_sets_to_dirtype(landscape, zoom_in, zoom_out) 

    print ("="*10, " Edit Images... ", "="*10)
    edit_all_images(landscape, "nose")
    edit_all_images(zoom_in, "eye")
    edit_all_images(zoom_out, "eye")

    print ("="*10, " Save Images... ", "="*10)
    save_images(input_path, "output", landscape, dir_path="landscape")
    save_images(input_path, "output", zoom_in, dir_path="zoom_in")
    save_images(input_path, "output", zoom_out, dir_path="zoom_out")

    print ("="*10, " Save Log file... ", "="*10)
    save_log(input_path, "output", [landscape, zoom_in, zoom_out])

    print ("@ Complete! @")
    return True
