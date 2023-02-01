# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import argparse
import pandas as pd


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        default="./output_pipeline",
        help="benchmark log path")
    parser.add_argument(
        "--output_name",
        type=str,
        default="benchmark_excel.xlsx",
        help="output excel file name")
    parser.add_argument(
        "--analysis_trt", dest="analysis_trt", action='store_true')
    parser.add_argument(
        "--analysis_mkl", dest="analysis_mkl", action='store_true')
    return parser.parse_args()


def find_all_logs(path_walk):
    """
    find all .log files from target dir
    """
    for root, ds, files in os.walk(path_walk):
        for file_name in files:
            if re.match(r'.*.log', file_name):
                full_path = os.path.join(root, file_name)
                yield file_name, full_path


def process_log(file_name):
    """
    process log to dict
    """
    output_dict = {}
    with open(file_name, 'r') as f:
        for i, data in enumerate(f.readlines()):
            if i == 0:
                continue
            line_lists = data.split(" ")

            # conf info
            if "runtime_device:" in line_lists:
                pos_buf = line_lists.index("runtime_device:")
                output_dict["runtime_device"] = line_lists[pos_buf + 1].strip()
            if "ir_optim:" in line_lists:
                pos_buf = line_lists.index("ir_optim:")
                output_dict["ir_optim"] = line_lists[pos_buf + 1].strip()
            if "enable_memory_optim:" in line_lists:
                pos_buf = line_lists.index("enable_memory_optim:")
                output_dict["enable_memory_optim"] = line_lists[pos_buf +
                                                                1].strip()
            if "enable_tensorrt:" in line_lists:
                pos_buf = line_lists.index("enable_tensorrt:")
                output_dict["enable_tensorrt"] = line_lists[pos_buf + 1].strip()
            if "precision:" in line_lists:
                pos_buf = line_lists.index("precision:")
                output_dict["precision"] = line_lists[pos_buf + 1].strip()
            if "enable_mkldnn:" in line_lists:
                pos_buf = line_lists.index("enable_mkldnn:")
                output_dict["enable_mkldnn"] = line_lists[pos_buf + 1].strip()
            if "cpu_math_library_num_threads:" in line_lists:
                pos_buf = line_lists.index("cpu_math_library_num_threads:")
                output_dict["cpu_math_library_num_threads"] = line_lists[
                    pos_buf + 1].strip()

            # model info
            if "model_name:" in line_lists:
                pos_buf = line_lists.index("model_name:")
                output_dict["model_name"] = list(
                    filter(None, line_lists[pos_buf + 1].strip().split('/')))[
                        -1]

            # data info
            if "batch_size:" in line_lists:
                pos_buf = line_lists.index("batch_size:")
                output_dict["batch_size"] = line_lists[pos_buf + 1].strip()
            if "input_shape:" in line_lists:
                pos_buf = line_lists.index("input_shape:")
                output_dict["input_shape"] = line_lists[pos_buf + 1].strip()

            # perf info
            if "cpu_rss(MB):" in line_lists:
                pos_buf = line_lists.index("cpu_rss(MB):")
                output_dict["cpu_rss(MB)"] = line_lists[pos_buf + 1].strip(
                ).split(',')[0]
            if "gpu_rss(MB):" in line_lists:
                pos_buf = line_lists.index("gpu_rss(MB):")
                output_dict["gpu_rss(MB)"] = line_lists[pos_buf + 1].strip(
                ).split(',')[0]
            if "gpu_util:" in line_lists:
                pos_buf = line_lists.index("gpu_util:")
                output_dict["gpu_util"] = line_lists[pos_buf + 1].strip().split(
                    ',')[0]
            if "preproce_time(ms):" in line_lists:
                pos_buf = line_lists.index("preproce_time(ms):")
                output_dict["preproce_time(ms)"] = line_lists[
                    pos_buf + 1].strip().split(',')[0]
            if "inference_time(ms):" in line_lists:
                pos_buf = line_lists.index("inference_time(ms):")
                output_dict["inference_time(ms)"] = line_lists[
                    pos_buf + 1].strip().split(',')[0]
            if "postprocess_time(ms):" in line_lists:
                pos_buf = line_lists.index("postprocess_time(ms):")
                output_dict["postprocess_time(ms)"] = line_lists[
                    pos_buf + 1].strip().split(',')[0]
    return output_dict


def filter_df_merge(cpu_df, filter_column=None):
    """
    process cpu data frame, merge by 'model_name', 'batch_size'
    Args:
        cpu_df ([type]): [description]
    """
    if not filter_column:
        raise Exception(
            "please assign filter_column for filter_df_merge function")

    df_lists = []
    filter_column_lists = []
    for k, v in cpu_df.groupby(filter_column, dropna=True):
        filter_column_lists.append(k)
        df_lists.append(v)
    final_output_df = df_lists[-1]

    # merge same model
    for i in range(len(df_lists) - 1):
        left_suffix = cpu_df[filter_column].unique()[0]
        right_suffix = df_lists[i][filter_column].unique()[0]
        print(left_suffix, right_suffix)
        if not pd.isnull(right_suffix):
            final_output_df = pd.merge(
                final_output_df,
                df_lists[i],
                how='left',
                left_on=['model_name', 'batch_size'],
                right_on=['model_name', 'batch_size'],
                suffixes=('', '_{0}_{1}'.format(filter_column, right_suffix)))

    # rename default df columns
    origin_column_names = list(cpu_df.columns.values)
    origin_column_names.remove(filter_column)
    suffix = final_output_df[filter_column].unique()[0]
    for name in origin_column_names:
        final_output_df.rename(
            columns={name: "{0}_{1}_{2}".format(name, filter_column, suffix)},
            inplace=True)
    final_output_df.rename(
        columns={
            filter_column: "{0}_{1}_{2}".format(filter_column, filter_column,
                                                suffix)
        },
        inplace=True)

    final_output_df.sort_values(
        by=[
            "model_name_{0}_{1}".format(filter_column, suffix),
            "batch_size_{0}_{1}".format(filter_column, suffix)
        ],
        inplace=True)
    return final_output_df


def trt_perf_analysis(raw_df):
    """
    sperate raw dataframe to a list of dataframe
    compare tensorrt percision performance
    """
    # filter df by gpu, compare tensorrt and gpu
    # define default dataframe for gpu performance analysis
    gpu_df = raw_df.loc[raw_df['runtime_device'] == 'gpu']
    new_df = filter_df_merge(gpu_df, "precision")

    # calculate qps diff percentile
    infer_fp32 = "inference_time(ms)_precision_fp32"
    infer_fp16 = "inference_time(ms)_precision_fp16"
    infer_int8 = "inference_time(ms)_precision_int8"
    new_df["fp32_fp16_diff"] = new_df[[infer_fp32, infer_fp16]].apply(
        lambda x: (float(x[infer_fp16]) - float(x[infer_fp32])) / float(x[infer_fp32]),
        axis=1)
    new_df["fp32_gpu_diff"] = new_df[["inference_time(ms)", infer_fp32]].apply(
        lambda x: (float(x[infer_fp32]) - float(x[infer_fp32])) / float(x["inference_time(ms)"]),
        axis=1)
    new_df["fp16_int8_diff"] = new_df[[infer_fp16, infer_int8]].apply(
        lambda x: (float(x[infer_int8]) - float(x[infer_fp16])) / float(x[infer_fp16]),
        axis=1)

    return new_df


def mkl_perf_analysis(raw_df):
    """
    sperate raw dataframe to a list of dataframe
    compare mkldnn performance with not enable mkldnn
    """
    # filter df by cpu, compare mkl and cpu
    # define default dataframe for cpu mkldnn analysis
    cpu_df = raw_df.loc[raw_df['runtime_device'] == 'cpu']
    mkl_compare_df = cpu_df.loc[cpu_df['cpu_math_library_num_threads'] == '1']
    thread_compare_df = cpu_df.loc[cpu_df['enable_mkldnn'] == 'True']

    # define dataframe need to be analyzed
    output_mkl_df = filter_df_merge(mkl_compare_df, 'enable_mkldnn')
    output_thread_df = filter_df_merge(thread_compare_df,
                                       'cpu_math_library_num_threads')

    # calculate performance diff percentile
    # compare mkl performance with cpu
    enable_mkldnn = "inference_time(ms)_enable_mkldnn_True"
    disable_mkldnn = "inference_time(ms)_enable_mkldnn_False"
    output_mkl_df["mkl_infer_diff"] = output_mkl_df[[
        enable_mkldnn, disable_mkldnn
    ]].apply(
        lambda x: (float(x[enable_mkldnn]) - float(x[disable_mkldnn])) / float(x[disable_mkldnn]),
        axis=1)
    cpu_enable_mkldnn = "cpu_rss(MB)_enable_mkldnn_True"
    cpu_disable_mkldnn = "cpu_rss(MB)_enable_mkldnn_False"
    output_mkl_df["mkl_cpu_rss_diff"] = output_mkl_df[[
        cpu_enable_mkldnn, cpu_disable_mkldnn
    ]].apply(
        lambda x: (float(x[cpu_enable_mkldnn]) - float(x[cpu_disable_mkldnn])) / float(x[cpu_disable_mkldnn]),
        axis=1)

    # compare cpu_multi_thread performance with cpu
    num_threads_1 = "inference_time(ms)_cpu_math_library_num_threads_1"
    num_threads_6 = "inference_time(ms)_cpu_math_library_num_threads_6"
    output_thread_df["mkl_infer_diff"] = output_thread_df[[
        num_threads_6, num_threads_1
    ]].apply(
        lambda x: (float(x[num_threads_6]) - float(x[num_threads_1])) / float(x[num_threads_1]),
        axis=1)
    cpu_num_threads_1 = "cpu_rss(MB)_cpu_math_library_num_threads_1"
    cpu_num_threads_6 = "cpu_rss(MB)_cpu_math_library_num_threads_6"
    output_thread_df["mkl_cpu_rss_diff"] = output_thread_df[[
        cpu_num_threads_6, cpu_num_threads_1
    ]].apply(
        lambda x: (float(x[cpu_num_threads_6]) - float(x[cpu_num_threads_1])) / float(x[cpu_num_threads_1]),
        axis=1)

    return output_mkl_df, output_thread_df


def main():
    """
    main
    """
    args = parse_args()
    # create empty DataFrame
    origin_df = pd.DataFrame(columns=[
        "model_name", "batch_size", "input_shape", "runtime_device", "ir_optim",
        "enable_memory_optim", "enable_tensorrt", "precision", "enable_mkldnn",
        "cpu_math_library_num_threads", "preproce_time(ms)",
        "inference_time(ms)", "postprocess_time(ms)", "cpu_rss(MB)",
        "gpu_rss(MB)", "gpu_util"
    ])

    for file_name, full_path in find_all_logs(args.log_path):
        dict_log = process_log(full_path)
        origin_df = origin_df.append(dict_log, ignore_index=True)

    raw_df = origin_df.sort_values(by='model_name')
    raw_df.sort_values(by=["model_name", "batch_size"], inplace=True)
    raw_df.to_excel(args.output_name)

    if args.analysis_trt:
        trt_df = trt_perf_analysis(raw_df)
        trt_df.to_excel("trt_analysis_{}".format(args.output_name))

    if args.analysis_mkl:
        mkl_df, thread_df = mkl_perf_analysis(raw_df)
        mkl_df.to_excel("mkl_enable_analysis_{}".format(args.output_name))
        thread_df.to_excel("mkl_threads_analysis_{}".format(args.output_name))


if __name__ == "__main__":
    main()
