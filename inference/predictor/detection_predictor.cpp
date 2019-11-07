// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "detection_predictor.h"
#include <cstring>
#include <cmath>
#include <fstream>
#include "utils/detection_result.pb.h"
#undef min

namespace PaddleSolution {
    /* lod_buffer: every item in lod_buffer is an image matrix after preprocessing
     * input_buffer: same data with lod_buffer after flattening to 1-D vector and padding, needed to be empty before using this function
     */
    void padding_minibatch(const std::vector<std::vector<float>> &lod_buffer,
                           std::vector<float> &input_buffer,
                           std::vector<int> &resize_heights,
                           std::vector<int> &resize_widths,
                           int channels, int coarsest_stride = 1) {
        int batch_size = lod_buffer.size();
        int max_h = -1;
        int max_w = -1;
        for (int i = 0; i < batch_size; ++i) {
            max_h = (max_h > resize_heights[i])? max_h:resize_heights[i];
            max_w = (max_w > resize_widths[i])? max_w:resize_widths[i];
        }
        max_h = static_cast<int>(ceil(static_cast<float>(max_h)
              / static_cast<float>(coarsest_stride)) * coarsest_stride);
        max_w = static_cast<int>(ceil(static_cast<float>(max_w)
              / static_cast<float>(coarsest_stride)) * coarsest_stride);
        std::cout << "max_w: " << max_w << " max_h: " << max_h << std::endl;
        input_buffer.insert(input_buffer.end(),
                            batch_size * channels * max_h * max_w, 0);
        // flatten tensor and padding
        for (int i = 0; i < lod_buffer.size(); ++i) {
            float *input_buffer_ptr = input_buffer.data()
                                    + i * channels * max_h * max_w;
            const float *lod_ptr = lod_buffer[i].data();
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < resize_heights[i]; ++h) {
                    memcpy(input_buffer_ptr, lod_ptr,
                           resize_widths[i] * sizeof(float));
                    lod_ptr += resize_widths[i];
                    input_buffer_ptr += max_w;
                }
                input_buffer_ptr += (max_h - resize_heights[i]) * max_w;
            }
        }
        // change resize w, h
        for (int i = 0; i < batch_size; ++i) {
            resize_widths[i] = max_w;
            resize_heights[i] = max_h;
        }
    }

    void output_detection_result(const float* out_addr,
                             const std::vector<std::vector<size_t>> &lod_vector,
                             const std::vector<std::string> &imgs_batch) {
        for (int i = 0; i < lod_vector[0].size() - 1; ++i) {
            DetectionResult detection_result;
            detection_result.set_filename(imgs_batch[i]);
            std::cout << imgs_batch[i] << ":" << std::endl;
            for (int j = lod_vector[0][i]; j < lod_vector[0][i+1]; ++j) {
                DetectionBox *box_ptr = detection_result.add_detection_boxes();
                box_ptr->set_class_(
                         static_cast<int>(round(out_addr[0 + j * 6])));
                box_ptr->set_score(out_addr[1 + j * 6]);
                box_ptr->set_left_top_x(out_addr[2 + j * 6]);
                box_ptr->set_left_top_y(out_addr[3 + j * 6]);
                box_ptr->set_right_bottom_x(out_addr[4 + j * 6]);
                box_ptr->set_right_bottom_y(out_addr[5 + j * 6]);
                printf("Class %d, score = %f, left top = [%f, %f], right bottom = [%f, %f]\n",
                        static_cast<int>(round(out_addr[0 + j * 6])),
                        out_addr[1 + j * 6],
                        out_addr[2 + j * 6],
                        out_addr[3 + j * 6],
                        out_addr[4 + j * 6],
                        out_addr[5 + j * 6]);
            }
            printf("\n");
            std::ofstream output(imgs_batch[i] + ".pb",
                    std::ios::out | std::ios::trunc | std::ios::binary);
            detection_result.SerializeToOstream(&output);
            output.close();
        }
    }

    int DetectionPredictor::init(const std::string& conf) {
        if (!_model_config.load_config(conf)) {
        #ifdef _WIN32
            std::cerr << "Fail to load config file: [" << conf << "], " 
                      << "please check whether the config file path is correct"
                      << std::endl;
        #else
            LOG(FATAL) << "Fail to load config file: [" << conf << "], "
                      << "please check whether the config file path is correct";
        #endif
            return -1;
        }
        _preprocessor = PaddleSolution::create_processor(conf);
        if (_preprocessor == nullptr) {
        #ifdef _WIN32
            std::cerr << "Failed to create_processor, please check whether you"
                      << " write a correct config file." << std::endl;
        #else
            LOG(FATAL) << "Failed to create_processor, please check whether"
                      << " you write a correct config file.";
        #endif
            return -1;
        }

        bool use_gpu = _model_config._use_gpu;
        const auto& model_dir = _model_config._model_path;
        const auto& model_filename = _model_config._model_file_name;
        const auto& params_filename = _model_config._param_file_name;

        // load paddle model file
        if (_model_config._predictor_mode == "NATIVE") {
            paddle::NativeConfig config;
            auto prog_file = utils::path_join(model_dir, model_filename);
            auto param_file = utils::path_join(model_dir, params_filename);
            config.prog_file = prog_file;
            config.param_file = param_file;
            config.fraction_of_gpu_memory = 0;
            config.use_gpu = use_gpu;
            config.device = 0;
            _main_predictor = paddle::CreatePaddlePredictor(config);
        } else if (_model_config._predictor_mode == "ANALYSIS") {
            paddle::AnalysisConfig config;
            if (use_gpu) {
                config.EnableUseGpu(100, 0);
            }
            auto prog_file = utils::path_join(model_dir, model_filename);
            auto param_file = utils::path_join(model_dir, params_filename);
            config.SetModel(prog_file, param_file);
            config.SwitchUseFeedFetchOps(false);
            config.SwitchSpecifyInputNames(true);
            config.EnableMemoryOptim();
            // config.SwitchIrOptim(true);
            // config.EnableTensorRtEngine(1<<4, 30, 3);
            _main_predictor = paddle::CreatePaddlePredictor(config);
        } else {
            return -1;
        }
        return 0;
    }

    int DetectionPredictor::predict(const std::vector<std::string>& imgs) {
        if (imgs.size() == 0) {
        #ifdef _WIN32
            std::cerr << "No image found! Please check whether the images path"
                      << " is correct or the format of images is correct\n"
                      << "Supporting format: [.jpeg|.jpg|.JPEG|.JPG|.bmp|.BMP|.png|.PNG]" << std::endl;
        #else
            LOG(ERROR) << "No image found! Please check whether the images path"
                       << " is correct or the format of images is correct\n"
                       << "Supporting format: [.jpeg|.jpg|.JPEG|.JPG|.bmp|.BMP|.png|.PNG]";
        #endif
            return -1;
        }
        if (_model_config._predictor_mode == "NATIVE") {
            return native_predict(imgs);
        } else if (_model_config._predictor_mode == "ANALYSIS") {
            return analysis_predict(imgs);
        }
        return -1;
    }

    int DetectionPredictor::native_predict(const std::vector<std::string>& imgs) {
        int config_batch_size = _model_config._batch_size;

        int channels = _model_config._channels;
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        std::size_t total_size = imgs.size();
        int default_batch_size = std::min(config_batch_size,
                                          static_cast<int>(total_size));
        int batch = total_size / default_batch_size +
                    ((total_size % default_batch_size) != 0);
        int batch_buffer_size = default_batch_size * channels
                              * eval_width * eval_height;

        auto& input_buffer = _buffer;
        auto& imgs_batch = _imgs_batch;
        float sr;
        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels
                                 * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            input_buffer.clear();
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }
            std::vector<int> ori_widths;
            std::vector<int> ori_heights;
            std::vector<int> resize_widths;
            std::vector<int> resize_heights;
            std::vector<float> scale_ratios;
            ori_widths.resize(batch_size);
            ori_heights.resize(batch_size);
            resize_widths.resize(batch_size);
            resize_heights.resize(batch_size);
            scale_ratios.resize(batch_size);
            std::vector<std::vector<float>> lod_buffer(batch_size);
            if (!_preprocessor->batch_process(imgs_batch, lod_buffer,
                                              ori_widths.data(),
                                              ori_heights.data(),
                                              resize_widths.data(),
                                              resize_heights.data(),
                                              scale_ratios.data())) {
                return -1;
            }
            // flatten and padding
            padding_minibatch(lod_buffer, input_buffer, resize_heights,
                              resize_widths, channels,
                              _model_config._coarsest_stride);
            paddle::PaddleTensor im_tensor, im_size_tensor, im_info_tensor;

            im_tensor.name = "image";
            im_tensor.shape = std::vector<int>({ batch_size,
                                                 channels,
                                                 resize_heights[0],
                                                 resize_widths[0] });
            im_tensor.data.Reset(input_buffer.data(),
                                 input_buffer.size() * sizeof(float));
            im_tensor.dtype = paddle::PaddleDType::FLOAT32;

            std::vector<float> image_infos;
            for (int i = 0; i < batch_size; ++i) {
                image_infos.push_back(resize_heights[i]);
                image_infos.push_back(resize_widths[i]);
                image_infos.push_back(scale_ratios[i]);
            }
            im_info_tensor.name = "info";
            im_info_tensor.shape = std::vector<int>({batch_size, 3});
            im_info_tensor.data.Reset(image_infos.data(),
                                      batch_size * 3 * sizeof(float));
            im_info_tensor.dtype = paddle::PaddleDType::FLOAT32;

            std::vector<int> image_size;
            for (int i = 0; i < batch_size; ++i) {
                image_size.push_back(ori_heights[i]);
                image_size.push_back(ori_widths[i]);
            }

           std::vector<float> image_size_f;
           for (int i = 0; i < batch_size; ++i) {
               image_size_f.push_back(ori_heights[i]);
               image_size_f.push_back(ori_widths[i]);
               image_size_f.push_back(1.0);
           }

           int feeds_size = _model_config._feeds_size;
           im_size_tensor.name = "im_size";
           if (feeds_size == 2) {
                im_size_tensor.shape = std::vector<int>({ batch_size, 2});
                im_size_tensor.data.Reset(image_size.data(),
                                          batch_size * 2 * sizeof(int));
                im_size_tensor.dtype = paddle::PaddleDType::INT32;
           } else if (feeds_size == 3) {
                im_size_tensor.shape = std::vector<int>({ batch_size, 3});
                im_size_tensor.data.Reset(image_size_f.data(),
                                          batch_size * 3 * sizeof(float));
                im_size_tensor.dtype = paddle::PaddleDType::FLOAT32;
           }
           std::cout << "Feed size = " << feeds_size << std::endl;
           feeds.push_back(im_tensor);
           if (_model_config._feeds_size > 2) {
                feeds.push_back(im_info_tensor);
           }
           feeds.push_back(im_size_tensor);
           _outputs.clear();

            auto t1 = std::chrono::high_resolution_clock::now();
            if (!_main_predictor->Run(feeds, &_outputs, batch_size)) {
            #ifdef _WIN32
                std::cerr << "Failed: NativePredictor->Run() return false at batch: " << u;
            #else            
                LOG(ERROR) << "Failed: NativePredictor->Run() return false at batch: " << u;
            #endif
                continue;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;
            std::cout << "Number of outputs:"  << _outputs.size() << std::endl;
            int out_num = 1;
            // print shape of first output tensor for debugging
            std::cout << "size of outputs[" << 0 << "]: (";
            for (int j = 0; j < _outputs[0].shape.size(); ++j) {
                out_num *= _outputs[0].shape[j];
                std::cout << _outputs[0].shape[j] << ",";
            }
            std::cout << ")" << std::endl;

        //    const size_t nums = _outputs.front().data.length() / sizeof(float);
        //    if (out_num % batch_size != 0 || out_num != nums) {
        //        LOG(ERROR) << "outputs data size mismatch with shape size.";
        //        return -1;
        //    }
            float* out_addr = reinterpret_cast<float *>(_outputs[0].data.data());
            output_detection_result(out_addr, _outputs[0].lod, imgs_batch);
        }
        return 0;
    }

    int DetectionPredictor::analysis_predict(
                const std::vector<std::string>& imgs) {
        int config_batch_size = _model_config._batch_size;
        int channels = _model_config._channels;
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        auto total_size = imgs.size();
        int default_batch_size = std::min(config_batch_size,
                                          static_cast<int>(total_size));
        int batch = total_size / default_batch_size
                + ((total_size % default_batch_size) != 0);
        int batch_buffer_size = default_batch_size * channels
                              * eval_width * eval_height;

        auto& input_buffer = _buffer;
        auto& imgs_batch = _imgs_batch;
        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels *
                                   eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            // input_buffer.resize(real_buffer_size);
            input_buffer.clear();
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }

            std::vector<int> ori_widths;
            std::vector<int> ori_heights;
            std::vector<int> resize_widths;
            std::vector<int> resize_heights;
            std::vector<float> scale_ratios;
            ori_widths.resize(batch_size);
            ori_heights.resize(batch_size);
            resize_widths.resize(batch_size);
            resize_heights.resize(batch_size);
            scale_ratios.resize(batch_size);

            std::vector<std::vector<float>> lod_buffer(batch_size);
            if (!_preprocessor->batch_process(imgs_batch, lod_buffer,
                                              ori_widths.data(),
                                              ori_heights.data(),
                                              resize_widths.data(),
                                              resize_heights.data(),
                                              scale_ratios.data())) {
                std::cout << "Failed to preprocess!" << std::endl;
                return -1;
            }

            // flatten tensor
            padding_minibatch(lod_buffer, input_buffer, resize_heights,
                              resize_widths, channels,
                              _model_config._coarsest_stride);

            std::vector<std::string> input_names = _main_predictor->GetInputNames();
            auto im_tensor = _main_predictor->GetInputTensor(
                                              input_names.front());
            im_tensor->Reshape({ batch_size, channels,
                                resize_heights[0], resize_widths[0] });
            im_tensor->copy_from_cpu(input_buffer.data());

            if (input_names.size() > 2) {
                std::vector<float> image_infos;
                for (int i = 0; i < batch_size; ++i) {
                    image_infos.push_back(resize_heights[i]);
                    image_infos.push_back(resize_widths[i]);
                    image_infos.push_back(scale_ratios[i]);
                }
                auto im_info_tensor = _main_predictor->GetInputTensor(
                                      input_names[1]);
                im_info_tensor->Reshape({batch_size, 3});
                im_info_tensor->copy_from_cpu(image_infos.data());
            }

            std::vector<int> image_size;
            for (int i = 0; i < batch_size; ++i) {
                image_size.push_back(ori_heights[i]);
                image_size.push_back(ori_widths[i]);
            }
            std::vector<float> image_size_f;
            for (int i = 0; i < batch_size; ++i) {
                image_size_f.push_back(static_cast<float>(ori_heights[i]));
                image_size_f.push_back(static_cast<float>(ori_widths[i]));
                image_size_f.push_back(1.0);
            }

            auto im_size_tensor = _main_predictor->GetInputTensor(
                                                    input_names.back());
            if (input_names.size() > 2) {
                im_size_tensor->Reshape({batch_size, 3});
                im_size_tensor->copy_from_cpu(image_size_f.data());
            } else {
                im_size_tensor->Reshape({batch_size, 2});
                im_size_tensor->copy_from_cpu(image_size.data());
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            _main_predictor->ZeroCopyRun();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;

            auto output_names = _main_predictor->GetOutputNames();
            auto output_t = _main_predictor->GetOutputTensor(output_names[0]);
            std::vector<float> out_data;
            std::vector<int> output_shape = output_t->shape();

            int out_num = 1;
            std::cout << "size of outputs[" << 0 << "]: (";
            for (int j = 0; j < output_shape.size(); ++j) {
                out_num *= output_shape[j];
                std::cout << output_shape[j] << ",";
            }
            std::cout << ")" << std::endl;

            out_data.resize(out_num);
            output_t->copy_to_cpu(out_data.data());

            float* out_addr = reinterpret_cast<float *>(out_data.data());
            auto lod_vector = output_t->lod();
            output_detection_result(out_addr, lod_vector, imgs_batch);
        }
        return 0;
    }
}  // namespace PaddleSolution
