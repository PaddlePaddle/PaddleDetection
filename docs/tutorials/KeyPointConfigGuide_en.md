**# config yaml guide**

KeyPoint config guide，Take an example of [tinypose_256x192.yml](../../configs/keypoint/tiny_pose/tinypose_256x192.yml)

```yaml
use_gpu: true                                                                                  #train with gpu or not

log_iter: 5                                                                                    #print log every 5 iter

save_dir: output                                                                               #the directory to save model

snapshot_epoch: 10                                                                             #save model every 10 epochs

weights: output/tinypose_256x192/model_final                                                   #the weight to load(without postfix “.pdparams”）

epoch: 420                                                                                     #the total epoch number to train

num_joints: &num_joints 17                                                                     #number of joints

pixel_std: &pixel_std 200                                                                      #the standard pixel length（don't care）

metric: KeyPointTopDownCOCOEval                                                                #metric function

num_classes: 1                                                                                 #number of classes（just for object detection, don't care）

train_height: &train_height 256                                                                #the height of model input

train_width: &train_width 192                                                                  #the width of model input

trainsize: &trainsize [*train_width, *train_height]                                            #the shape of model input

hmsize: &hmsize [48, 64]                                                                       #the shape of model output

flip_perm: &flip_perm [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]  #the correspondence between left and right keypoint id, for example: left wrist become right wrist after image flip, and also the right wrist becomes left wrist





\#####model

architecture: TopDownHRNet                                                                     #the model architecture



TopDownHRNet:                                                                                  #TopDownHRNet configs

  backbone: LiteHRNet                                                                          #which backbone to use

  post_process: HRNetPostProcess                                                               #the post_process to use

  flip_perm: *flip_perm                                                                        #same to the upper "flip_perm"

  num_joints: *num_joints                                                                      #the joint number（the number of output channels）

  width: &width 40                                                                             #backbone output channels

  loss: KeyPointMSELoss                                                                        #loss funciton

  use_dark: true                                                                               #whther to use DarkPose in postprocess



LiteHRNet:                                                                                     #LiteHRNet configs

  network_type: wider_naive                                                                    #the network type of backbone

  freeze_at: -1                                                                                #the branch match this id doesn't backward，-1 means all branch backward

  freeze_norm: false                                                                           #whether to freeze normalize weights

  return_idx: [0]                                                                              #the branch id to fetch features



KeyPointMSELoss:                                                                               #Loss configs

  use_target_weight: true                                                                      #whether to use target weights

  loss_scale: 1.0                                                                              #loss weights，finalloss = loss*loss_scale



\#####optimizer

LearningRate:                                                                                  #LearningRate configs

  base_lr: 0.002                                                                               #the original base learning rate

  schedulers:

  \- !PiecewiseDecay                                                                           #the scheduler to adjust learning rate

​    milestones: [380, 410]                                                                     #the milestones(epochs) to adjust learning rate

​    gamma: 0.1                                                                                 #the ratio to adjust learning rate, new_lr = lr*gamma

  \- !LinearWarmup                                                                             #Warmup configs

​    start_factor: 0.001                                                                        #the original ratio with respect to base_lr

​    steps: 500                                                                                 #iters used to warmup



OptimizerBuilder:                                                                              #Optimizer type configs

  optimizer:

​    type: Adam                                                                                 #optimizer type: Adam

  regularizer:

​    factor: 0.0                                                                                #the regularizer weight

​    type: L2                                                                                   #regularizer type: L2/L1





\#####data

TrainDataset:                                                                                  #Train Dataset configs

  !KeypointTopDownCocoDataset                                                                  #the dataset class to load data

​    image_dir: ""                                                                              #the image directory, relative to dataset_dir

​    anno_path: aic_coco_train_cocoformat.json                                                  #the train datalist，coco format, relative to dataset_dir

​    dataset_dir: dataset                                                                       #the dataset directory, the image_dir and anno_path based on this directory

​    num_joints: *num_joints                                                                    #joint numbers

​    trainsize: *trainsize                                                                      #the input size of model

​    pixel_std: *pixel_std                                                                      #same to the upper "pixel_std"

​    use_gt_bbox: True                                                                          #whether to use gt bbox, commonly used in eval





EvalDataset:                                                                                   #Eval Dataset configs

  !KeypointTopDownCocoDataset                                                                  #the dataset class to load data

​    image_dir: val2017                                                                         #the image directory, relative to dataset_dir

​    anno_path: annotations/person_keypoints_val2017.json                                       #the eval datalist，coco format, relative to dataset_dir

​    dataset_dir: dataset/coco                                                                  #the dataset directory, the image_dir and anno_path based on this directory

​    num_joints: *num_joints                                                                    #joint numbers

​    trainsize: *trainsize                                                                      #the input size of model

​    pixel_std: *pixel_std                                                                      #same to the upper "pixel_std"

​    use_gt_bbox: True                                                                          #whether to use gt bbox, commonly used in eval

​    image_thre: 0.5                                                                            #the threshold of detected rect, used while use_gt_bbox is False



TestDataset:                                                                                   #the test dataset without label

  !ImageFolder                                                                                 #the class to load data, find images by folder

​    anno_path: dataset/coco/keypoint_imagelist.txt                                             #the image list file



worker_num: 2                                                                                  #the workers to load Dataset

global_mean: &global_mean [0.485, 0.456, 0.406]                                                #means used to normalize image

global_std: &global_std [0.229, 0.224, 0.225]                                                  #stds used to normalize image

TrainReader:                                                                                   #TrainReader configs

  sample_transforms:                                                                           #transform configs

​    \- RandomFlipHalfBodyTransform:                                                            #random flip & random HalfBodyTransform

​        scale: 0.25                                                                            #the maximum scale for size transform

​        rot: 30                                                                                #the maximum rotation to transoform

​        num_joints_half_body: 8                                                                #the HalfBodyTransform is skiped while joints found is less than this number

​        prob_half_body: 0.3                                                                    #the ratio of halfbody transform

​        pixel_std: *pixel_std                                                                  #same to upper "pixel_std"

​        trainsize: *trainsize                                                                  #the input size of model

​        upper_body_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                     #the joint id which is  belong to upper body

​        flip_pairs: *flip_perm                                                                 #same to the upper "flip_perm"

​    \- AugmentationbyInformantionDropping:

​        prob_cutout: 0.5                                                                       #the probability to cutout keypoint

​        offset_factor: 0.05                                                                    #the jitter offset of cutout position, expressed as a percentage of trainwidth

​        num_patch: 1                                                                           #the numbers of area to cutout

​        trainsize: *trainsize                                                                  #same to upper "trainsize"

​    \- TopDownAffine:

​        trainsize: *trainsize                                                                  #same to upper "trainsize"

​        use_udp: true                                                                          #whether to use udp_unbias（just for flip eval）

​    \- ToHeatmapsTopDown_DARK:                                                                 #generate gt heatmaps

​        hmsize: *hmsize                                                                        #the size of output heatmaps

​        sigma: 2                                                                               #the sigma of gaussin kernel which used to generate gt heatmaps

  batch_transforms:

​    \- NormalizeImage:                                                                         #image normalize class

​        mean: *global_mean                                                                     #mean of normalize

​        std: *global_std                                                                       #std of normalize

​        is_scale: true                                                                         #whether scale by 1/255 to every image pixels，transform pixel from [0,255] to [0,1]

​    \- Permute: {}                                                                             #channel transform from HWC to CHW

  batch_size: 128                                                                              #batchsize used for train

  shuffle: true                                                                                #whether to shuffle the images before train

  drop_last: false                                                                             #whether drop the last images which is not enogh for batchsize



EvalReader:

  sample_transforms:                                                                           #transform configs

​    \- TopDownAffine:                                                                          #Affine configs

​        trainsize: *trainsize                                                                  #same to upper "trainsize"

​        use_udp: true                                                                          #whether to use udp_unbias（just for flip eval）

  batch_transforms:

​    \- NormalizeImage:                                                                         #image normalize, the values should be same to values in TrainReader

​        mean: *global_mean

​        std: *global_std

​        is_scale: true

​    \- Permute: {}                                                                             #channel transform from HWC to CHW

  batch_size: 16                                                                               #batchsize used for test



TestReader:

  inputs_def:

​    image_shape: [3, *train_height, *train_width]                                              #the input dimensions used in model，CHW

  sample_transforms:

​    \- Decode: {}                                                                              #load image

​    \- TopDownEvalAffine:                                                                      #Affine class used in Eval

​        trainsize: *trainsize                                                                  #the input size of model

​    \- NormalizeImage:                                                                         #image normalize, the values should be same to values in TrainReader

​        mean: *global_mean                                                                     #mean of normalize

​        std: *global_std                                                                       #std of normalize

​        is_scale: true                                                                         #whether scale by 1/255 to every image pixels，transform pixel from [0,255] to [0,1]

​    \- Permute: {}                                                                             #channel transform from HWC to CHW

  batch_size: 1                                                                                #Test batchsize

  fuse_normalize: false                                                                        #whether fuse the normalize into model while export model, this speedup the model infer
```
