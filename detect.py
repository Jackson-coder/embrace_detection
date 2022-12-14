import argparse
import logging
import time
from pathlib import Path

import os
import copy
from urllib.parse import _NetlocResultMixinBase
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.yolo import Model
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, intersect_dicts

logger = logging.getLogger(__name__)


def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py, _ = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1, _ = corner
        x2, y2, _ = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1e-10)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def detect(opt, frame_filter=10, warning_frame=1):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    frame_id = 0
    warning_buffer = [0] * frame_filter
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label, nc=model.yaml['nc'], multi_cls_offset=True)

        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                if kpt_label:
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)
                    
                output = np.array([out.cpu().numpy() for out in det])
                bboxes = output[:, :4]
                poses = output[:, 6:].reshape((-1,17,3))

                flag = False
                
                for ii, (bbox_i, pose_i) in enumerate(zip(bboxes, poses)):
                    for jj, (bbox_j, pose_j) in enumerate(zip(bboxes, poses)):  # i 拥抱 j
                        if ii != jj and abs(bbox_i[3]-bbox_i[1]) / abs(bbox_j[3]-bbox_j[1]) > 0.8 and \
                            abs(bbox_i[3]-bbox_i[1]) / abs(bbox_j[3]-bbox_j[1]) < 1.25:
                                if abs((pose_i[9][1]-pose_i[7][1])/abs(pose_i[9][0]-pose_i[7][0]+1e-10)) > 2 and abs((pose_i[5][1]-pose_i[7][1])/abs(pose_i[5][0]-pose_i[7][0]+1e-10)) > 2 and\
                                    abs((pose_i[10][1]-pose_i[8][1])/abs(pose_i[10][0]-pose_i[8][0]+1e-10)) > 2 and abs((pose_i[6][1]-pose_i[8][1])/abs(pose_i[6][0]-pose_i[8][0]+1e-10)) > 2:
                                    continue
                                if (pose_i[5][0]-pose_j[0][0])*(pose_i[9][0]-pose_j[0][0]) < 0:   # 手腕
                                    # cv2.circle(im0, (int(pose_i[9][0]), int(pose_i[9][1])), 10, (0, 0, 255), 8)
                                    flag = True
                                elif (pose_i[6][0]-pose_j[0][0])*(pose_i[10][0]-pose_j[0][0]) < 0:
                                    # cv2.circle(im0, (int(pose_i[10][0]), int(pose_i[10][1])), 10, (0, 0, 255), 8)
                                    flag = True
                                elif (pose_i[5][0]-pose_j[0][0])*(pose_i[7][0]-pose_j[0][0]) < 0:  # 手肘
                                    # cv2.circle(im0, (int(pose_i[7][0]), int(pose_i[7][1])), 10, (0, 0, 255), 8)
                                    flag = True
                                elif (pose_i[6][0]-pose_j[0][0])*(pose_i[8][0]-pose_j[0][0]) < 0:
                                    # cv2.circle(im0, (int(pose_i[8][0]), int(pose_i[8][1])), 10, (0, 0, 255), 8)
                                    flag = True                    
                
                warning_buffer[frame_id%frame_filter]=1 if flag else 0
                isWarning = sum(warning_buffer)>warning_frame
                frame_id += 1
                                        
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        kpts = det[det_index, 6:]
                        # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


                if save_txt_tidl:  # Write to file in tidl dump format
                    for *xyxy, conf, cls in det_tidl:
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if isWarning:
                        im0 = cv2.putText(im0, 'Warning', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    vid_writer.write(im0)

    if save_txt or save_txt_tidl or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt or save_txt_tidl else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

# action_cls = {"sitting":0,"standing":1,"lying":2,"other":3}

# 全员坐满进行初始化
def init(detect_results):
    center_x = (detect_results[..., 0] + detect_results[..., 2]) / 2
    center_y = (detect_results[..., 1] + detect_results[..., 3]) / 2
    w = abs(detect_results[..., 0] - detect_results[..., 2])
    h = abs(detect_results[..., 1] - detect_results[..., 3])
    state = np.zeros(len(detect_results)) # 躺坐为0， 离席为1
    seat = np.concatenate((center_x,center_y,w,h,seat,state.T),axis=1)
    return seat


def warningAction(detect_result, conf, seat, sigma=1):
    if detect_result[-2] < conf:
        return
    distance = abs(seat[:,0]-detect_result[0]) + abs(seat[:,1]-detect_result[1])
    min_index = distance.argmin()
    min_value = distance.min()
    if min_value < (seat[min_index][2]+seat[min_index][3])/sigma:
        if detect_result[-1] == 2:
            seat[min_index][-1] = 0
            return "lying"
        elif detect_result[-1] == 0:
            seat[min_index][-1] = 0
            return "sitting"
        elif detect_result[-1] == 1:
            seat[min_index][-1] = 1
            return "standing"
    else:
        return 

    
def countAndAction(detect_results, conf, seat, sigma=1):
    seat_number = seat.shape[0]
    person_number = 0
    detect_lying = []
    for detect_result in detect_results:
        state = warningAction(detect_result, conf, seat, sigma=sigma)
        if state is not None:
            person_number += 1
        if state == "lying":
            detect_lying.append(detect_result)
    
    flag = True if person_number < seat_number else False
    
    # 趴桌(人)， 缺席（状态值）, 离席（座位，可以通过最有一个元素查看离席的座位）
    return detect_lying, flag, seat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/hug', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size',  type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint labels')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
