import os
import cv2
import paddle
import argparse
import numpy as np

from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose


default_path = './weights/ReXNetV1-size-256-wingloss102-0.122.pdparams'


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description=' camera ')
    parser.add_argument('--model_path' , type=str  , default = default_path, help = 'model_path' )   
    parser.add_argument('--model'      , type=str  , default = 'ReXNetV1'  , help = 'model'      )        
    parser.add_argument('--num_classes', type=int  , default = 42          , help = 'num_classes')  
    parser.add_argument('--GPUS'       , type=str  , default = '0'         , help = 'GPUS'       )         
    parser.add_argument('--img_size'   , type=tuple, default = (256,256)   , help = 'img_size'   )     
    parser.add_argument('--camera_id'  , type=int  , default = 0           , help = 'camera id'  )
    parser.add_argument('--fps'        , type=int  , default = 30           , help = 'fps'       )
    ops = parser.parse_args()   
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    
    if ops.model == "ReXNetV1":
        model_ = ReXNetV1(num_classes=ops.num_classes)
    
    if os.access(ops.model_path,os.F_OK):
        chkpt = paddle.load(ops.model_path)
        model_.set_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))
        
    model_.eval()
    
    capture = cv2.VideoCapture(ops.camera_id)
    
    with paddle.no_grad():
        
        while 1:
            ret, img = capture.read()

            img_width = img.shape[1]
            img_height = img.shape[0]
            
            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.
            img_ = img_.transpose(2, 0, 1)
            img_ = paddle.to_tensor(img_)
            img_ = img_.unsqueeze_(0)
            
            pre_ = model_(img_)
            output = pre_.numpy()
            output = np.squeeze(output)
            
            pts_hand = {}
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))
        
                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
                
            draw_bd_handpose(img, pts_hand, 0, 0)
            
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))
        
                cv2.circle(img, (int(x),int(y)), 3, (255,50,60),-1)
                cv2.circle(img, (int(x),int(y)), 1, (255,150,180),-1)
                
                
            cv2.imshow("capture", img)
            

            if cv2.waitKey(1000//ops.fps) == ord('q'):          
                break
            
    capture.release()
    cv2.destroyAllWindows()