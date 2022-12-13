# YOLO-Pose Multi-person Pose estimation model
This repository is the official implementation of the paper ["**YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss**"](https://arxiv.org/abs/2204.06806) , accepted at Deep Learning for Efficient Computer Vision (ECV) workshop
at CVPR 2022. This repository contains YOLOv5 based models for human pose estimation.



## **YOLO-Pose Models and Ckpts**.

|Dataset | Model Name                                                                                                                                                                                                                                              |Input Size |GMACS  |AP[0.5:0.95]%| AP50%|Notes |
|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|----------|-------------|------|----- |
|COCO    | [Yolov5s6_pose_640](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_640_60p7_85p3_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |640x640    |**10.2**  |   57.5      | 84.3 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_640_60p7_85p3_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_640_60p7_85p3_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_960_71p6_93p1/weights/last.pt)|
|COCO    | [Yolov5s6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_960_64p8_87p4_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |960x960    |**22.8**  |   63.7      | 87.6 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_960_64p8_87p4_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_960_64p8_87p4_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_960_71p6_93p1/weights/last.pt)|
|COCO    | [Yolov5m6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5m6_960_67p8_89p3_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |960x960    |**66.3**  |   67.4      | 89.1 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5m6_960_67p8_89p3_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5m6_960_67p8_89p3_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5m6_960_74p1_93p6/weights/last.pt)|
|COCO    | [Yolov5l6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |960x960    |**145.6** |   69.4      | 90.2 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5l6_960_74p7_94p0/weights/last.pt)|



## **Training: YOLO-Pose**
Train a suitable model  by running the following command using a suitable pretrained ckpt from the previous section.

```
python train.py --data coco_kpts.yaml --cfg yolov5s6_kpts.yaml --weights 'path to the pre-trained ckpts' --batch-size 64 --img 960 --kpt-label
                                      --cfg yolov5m6_kpts.yaml 
                                      --cfg yolov5l6_kpts.yaml 
```


 TO train a model at different at input resolution of 640, run the command below:
```
python train.py --data coco_kpts.yaml --cfg yolov5s6_kpts.yaml --weights 'path to the pre-trained ckpts' --batch-size 64 --img 640 --kpt-label
```


## **YOLOv5-ti-lite Based Models and Ckpts**
This is a lite version of the the model as described here. These models will run efficiently on TI processors.

|Dataset |Model Name                      |Input Size |GMACS  |AP[0.5:0.95]%| AP50%|Notes |
|--------|------------------------------- |-----------|----------|-------------|------|----- |
|COCO    |[Yolov5s6_pose_640_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_640_ti_lite_54p9_82p2/weights/last.pt)     |640x640    |**8.6**  |  54.9      | 82.2 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_640_ti_lite_54p9_82p2/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_640_ti_lite_54p9_82p2/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_ti_lite_person_64p8_90p2/weights/last.pt)|
|COCO    |[Yolov5s6_pose_960_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_960_ti_lite_59p7_85p6/weights/last.pt)     |960x960    |**19.3** |  59.7      | 85.6 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_960_ti_lite_59p7_85p6/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_960_ti_lite_59p7_85p6/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_ti_lite_person_64p8_90p2/weights/last.pt)|
|COCO    |[Yolov5s6_pose_1280_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_1280_ti_lite_60p9_85p9/weights/last.pt)   |1280x1280  |**34.4** |  60.9      | 85.9 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_1280_ti_lite_60p9_85p9/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_1280_ti_lite_60p9_85p9/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_ti_lite_person_64p8_90p2/weights/last.pt)|
|COCO    |[Yolov5m6_pose_640_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5m6_640_ti_lite_60p5_86p8/weights/best.pt)     |640x640    |**26.1** |  60.5      | 86.8 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5m6_640_ti_lite_60p5_86p8/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5m6_640_ti_lite_60p5_86p8/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5m6_ti_lite_person_71p4_93p1/weights/last.pt)|
|COCO    |[Yolov5m6_pose_960_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5m6_960_ti_lite_65p9_88p6/weights/last.pt)     |960x960    |**58.7** |  65.9      | 88.6 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5m6_960_ti_lite_65p9_88p6/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5m6_960_ti_lite_65p9_88p6/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5m6_ti_lite_person_71p4_93p1/weights/last.pt)|

## **Training: YOLO-Pose-ti-lite**
Train a suitable model  by running the following command:

```
python train.py --data coco_kpts.yaml --cfg yolov5s6_kpts_ti_lite.yaml --weights 'path to the pre-trained ckpts' --batch-size 64 --img 960 --kpt-label --hyp hyp.scratch_lite.yaml
                                      --cfg yolov5m6_kpts_ti_lite.yaml 
                                      --cfg yolov5l6_kpts_ti_lite.yaml 
```
TO train a model at different at input resolution of 640, run the command below:
```
python train.py --data coco_kpts.yaml --cfg yolov5s6_kpts_ti_lite.yaml --weights 'path to the pre-trained ckpts' --batch-size 64 --img 640 --kpt-label --hyp hyp.scratch_lite.yaml
```

 The same pretrained model can be used here as well.


## **Model Inferring**

* Run the following command:
    ```
    python detect.py --device 3 --weights "path to the pre-trained ckpt" --img-size 960 --kpt-label
    ```
Recommand: [Yolov5l6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt)
<br/> 

###  **ONNX Export Including Detection and Pose Estimation:**
* Run the following command to export the entire models including the detection part, 
    ``` 
    python export.py --weights "path to the pre-trained ckpt"  --img 640 --batch 1 --simplify --export-nms # export at 640x640 with batch size 1
    ```
* Apart from exporting the complete ONNX model, above script will generate a prototxt file that contains information of the detection layer. This prototxt file is required to deploy the moodel on TI SoC.

###  **ONNXRT Inference: Human Pose Estimation Inference with an End-to-End ONNX Model:**

 * If you haven't exported a model with the above command, download a sample model from this [link](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/latest/edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx).
 * Run the script as below to run inference with an ONNX model. The script runs inference and visualize the results. There is no extra post-processing required. The ONNX model is self-sufficient unlike existing bottom-up approaches. The [script](onnx_inference/yolo_pose_onnx_inference.py) is compleletey independent and contains all perprocessing and visualization. 
    ``` 
    cd onnx_inference
    python yolo_pose_onnx_inference.py --model-path "path_to_onnx_model"  --img-path "sample_ips.txt" --dst-path "sample_ops_onnxrt"  # Run inference on a set of sample images as specified by sample_ips.txt
    ```
    

## **References**

[1] [Official YOLOV5 repository](https://github.com/ultralytics/yolov5/) <br>
[2] [yolov5-improvements-and-evaluation, Roboflow](https://blog.roboflow.com/yolov5-improvements-and-evaluation/) <br>
[3] [Focus layer in YOLOV5]( https://github.com/ultralytics/yolov5/discussions/3181) <br>
[4] [CrossStagePartial Network](https://github.com/WongKinYiu/CrossStagePartialNetworkss)  <br>
[5] Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. [CSPNet: A new backbone that can enhance learning capability of
cnn](https://arxiv.org/abs/1911.11929). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop (CVPR Workshop),2020. <br>
[6]Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. [Path aggregation network for instance segmentation](https://arxiv.org/abs/1803.01534). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 8759–8768, 2018 <br>
[7] [Efficientnet-lite quantization](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) <br>
[8] [YOLOv5 Training video from Texas Instruments](https://training.ti.com/process-efficient-object-detection-using-yolov5-and-tda4x-processors) <br> 
[9] [YOLO-Pose Training video from Texas Instruments:Upcoming](Upcoming)