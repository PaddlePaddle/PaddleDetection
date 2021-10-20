//
// Create by RangiLyu
// 2021 / 1 / 12
//

#ifndef _NANODET_OPENVINO_H_
#define _NANODET_OPENVINO_H_

#include <string>
#include <opencv2/core.hpp>
#include <inference_engine.hpp>


typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class PicoDet
{
public:
    PicoDet(const char* param);

    ~PicoDet();

    InferenceEngine::ExecutableNetwork network_;
    InferenceEngine::InferRequest infer_request_;
    // static bool hasGPU;

    std::vector<HeadInfo> heads_info_{
        // cls_pred|dis_pred|stride
        {"save_infer_model/scale_0.tmp_1", "save_infer_model/scale_4.tmp_1", 8},
        {"save_infer_model/scale_1.tmp_1", "save_infer_model/scale_5.tmp_1", 16},
        {"save_infer_model/scale_2.tmp_1", "save_infer_model/scale_6.tmp_1", 32},
        {"save_infer_model/scale_3.tmp_1", "save_infer_model/scale_7.tmp_1", 64},
    };

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

private:
    void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);
    void decode_infer(const float*& cls_pred, const float*& dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    std::string input_name_;
    int input_size_ = 320;
    int num_class_ = 80;
    int reg_max_ = 7;

};


#endif //_NANODE_TOPENVINO_H_
