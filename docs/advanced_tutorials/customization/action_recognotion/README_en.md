[简体中文](./README.md) | English

# Secondary Development for Action Recognition Task

In the process of industrial implementation, the application of action recognition algorithms will inevitably lead to the need for customized types of action, or the optimization of existing action recognition models to improve the performance of the model in specific scenarios. In view of the diversity of behaviors, PP-Human supports the identification of five abnormal behavioras of smoking, making phone calls, falling, fighting, and people intrusion. At the same time, according to the different behaviors, PP-Human integrates five action recognition technology solutions based on video classification, detection-based, image-based classification, tracking-based and skeleton-based, which can cover 90%+ action type recognition and meet various development needs. In this document, we use a case to introduce how to select a action recognition solution according to the expected behavior, and use PaddleDetection to carry out the secondary development of the action recognition algorithm, including: solution selection, data preparation, model optimization and development process for adding new actions.


## Solution Selection
In PaddleDetection's PP-Human, we provide a variety of solutions for behavior recognition: video classification, image classification, detection, tracking-based, and skeleton point-based behavior recognition solutions, in order to meet the needs of different scenes and different target behaviors.

<img width="1091" alt="image" src="https://user-images.githubusercontent.com/22989727/178742352-d0c61784-3e93-4406-b2a2-9067f42cb343.png">

The following takes several specific actions that PaddleDetection currently supports as an example to introduce the selection basis of each action:

### Smoking

Solution selection: action recognition based on detection with human id.

Reason: The smoking action has a obvious feature target, that is, cigarette. So we can think that when a cigarette is detected in the corresponding image of a person, the person is with the smoking action. Compared with video-based or skeleton-based recognition schemes, training detection model needs to collect data at the image level rather than the video level, which can significantly reduce the difficulty of data collection and labeling. In addition, the detection task has abundant pre-training model resources, and the performance of the model will be more guaranteed.

### Making Phone Calls

Solution selection: action recognition based on classification with human id.

Reason: Although there is a characteristic target of a mobile phone in the call action, in order to distinguish actions such as looking at the mobile phone, and considering that there will be much occlusion of the mobile phone in the calling action in the security scene (such as the occlusion of the mobile phone by the hand or head, etc.), is not conducive to the detection model to correctly detect the target. Simultaneous, calls usually last a long time, and the character's action do not change much, so a strategy for frame-level image classification can therefore be employed. In addition, the action of making a phone call can mainly be judged by the upper body, and the half-body picture can be used to remove redundant information to reduce the difficulty of model training.


### Falling

Solution selection: action recognition based on skelenton.

Reason: Falling is an obvious temporal action, which is distinguishable by a character himself, and it is scene-independent. Since PP-Human is towards the security monitoring scene, where the background changes are more complicated, and the real-time inference needs to be considered in the deployment, the action recognition based on skeleton points is adopted to obtain better generalization and running speed.


### People Intrusion

Solution selection: action recognition based on tracking with human id.

Reason: The intrusion recognition can be judged by whether the pedestrian's path or location is in a selected area, and it is unrelated to pedestrian's body action. Therefore, it is only necessary to track the human and use coordinate results to analyze whether there is intrusion behavior.

### Fighting

Solution selection: action recognition based on video classification.

Reason: Unlike the actions above, fighting is a typical multiplayer action. Therefore, the detection and tracking model is no longer used to extract pedestrians and their IDs, but the entire video clip is processed. In addition, the mutual occlusion between various targets in the fighting scene is extremely serious, leading to the accuracy of keypoint recognition is not good.



The following are detailed description for the five major categories of solutions, including the data preparation, model optimization and adding new actions.

1. [action recognition based on detection with human id.](./idbased_det_en.md)
2. [action recognition based on classification with human id.](./idbased_clas_en.md)
3. [action recognition based on skelenton.](./skeletonbased_rec_en.md)
4. [action recognition based on tracking with human id](../pphuman_mot_en.md)
5. [action recognition based on video classification](./videobased_rec_en.md)
