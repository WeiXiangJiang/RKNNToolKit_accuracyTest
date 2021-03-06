// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <string>
#include <iomanip>
#include <dirent.h>
#include <functional>
#include <unistd.h>
#include <sstream>
#include <algorithm>
#include <assert.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

std::vector <std::string> read_directory( const std::string& path = std::string() )
{
    std::vector <std::string> result;
    dirent* de;
    DIR* dp;
    errno = 0;
    dp = opendir( path.empty() ? "." : path.c_str() );
    if (dp)
    {
        while (true)
        {
            errno = 0;
            de = readdir( dp );
            if (de == NULL) break;
            if (strcmp(".", de->d_name) == 0 || strcmp("..", de->d_name) == 0)
                continue;
            if (std::string( de->d_name ).find(".JPEG") == string::npos && std::string( de->d_name ).find(".jpg") == string::npos)
                continue;
            result.push_back( std::string( de->d_name ) );
        }           
        closedir( dp );
        std::sort( result.begin(), result.end() );
    }
    return result;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}

static int rknn_GetTop
    (
    float *pfProb,
    float *pfMaxProb,
    uint32_t *pMaxClass,
    uint32_t outputCount,
    uint32_t topNum
    )
{
    uint32_t i, j;

    #define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM) return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i=0; i<outputCount; i++)
        {
            if ((i == *(pMaxClass+0)) || (i == *(pMaxClass+1)) || (i == *(pMaxClass+2)) ||
                (i == *(pMaxClass+3)) || (i == *(pMaxClass+4)))
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb+j))
            {
                *(pfMaxProb+j) = pfProb[i];
                *(pMaxClass+j) = i;
            }
        }
    }

    return 1;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int MODEL_IN_WIDTH = 112;
    const int MODEL_IN_HEIGHT = 112;
    const int MODEL_IN_CHANNELS = 3;
    std::string image_dir="./images/val/";
    std::string save_file="./result/result.txt";
    std::string list_name="imageslist.txt";
    int image_count = 0;
    int repeat_count = 500000;
    rknn_context ctx;
    int ret;
    int res;
    int model_len = 0;
    unsigned char *model;
    std::string model_file="./models/AT/SqueezeNet1.0-0000.params";
    int one_pic_repeat_count = 1;
    const char* one_pic_repeat = std::getenv("ONE_PIC_REPEAT_COUNT");
    if(one_pic_repeat)
        one_pic_repeat_count = std::strtoul(one_pic_repeat, NULL, 10);

    if(argc == 1)
    {
        std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                  << "[-m model_file] [-i image_dir] [-o save_file] [-l list_name] [-r repeat_count]\n"
                  << " \n";
        return 0;
    }
    while((res = getopt(argc, argv, "m:i:o:r:l:h")) != -1)
    {
        switch(res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_dir = optarg;
                break;
           case 'o':
                save_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            case 'l':
                list_name = optarg;
                break;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                          << "[-m model_file] [-i image_dir] [-o save_file] [-r repeat_count]  [-l list_name]\n"
                          << "\n";
                return 0;
            default:
                break;
        }
    }
    // Load RKNN Model
    model = load_model(model_file.c_str(), &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    // Load image
    std::string image_file;
    std::string lineStr;
    int i=0;
    int j;
    char* end;
    int top1_count = 0;
    int top5_count = 0;


    std::vector<std::string> img_list;
    std::ifstream list_stream(list_name);
    if (!list_stream.is_open())
    {
        fprintf(stderr, "Open image list failed.\n");
        return -1;
    }

    std::string image_name;
    while (std::getline(list_stream, image_name))
    {
        img_list.push_back(image_name);
    }

    // std::vector <std::string> img_list=read_directory(image_dir);
    std::ofstream feature_file(save_file);
    for (std::string image :img_list)
    {
        image_count = image_count + 1;
        if (image_count > repeat_count)
        {
            image_count=image_count-1;
            break;
        }
        
        std::cout << "test image count: " << image_count << "\n";
        image_file=(std::string(image_dir)+std::string(image));
        std::cout << image_file.c_str() << "\n";
        cv::Mat orig_img = imread(image_file, cv::IMREAD_COLOR);
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", image_file);
            return -1;
        }

        cv::Mat img = orig_img.clone();
        if(orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
            printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
            cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
        }

        cv::cvtColor(img, img, COLOR_BGR2RGB);
        // Set Input Data
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = img.cols*img.rows*img.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = img.data;

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        struct timeval t0, t1;
        float avg_time = 0.f;
        float min_time = __DBL_MAX__;
        float max_time = -__DBL_MAX__;
        for (int e = 0 ; e < one_pic_repeat_count; e++)
        {
            gettimeofday(&t0, NULL);
            // Run
            printf("rknn_run\n");
            ret = rknn_run(ctx, nullptr);
            gettimeofday(&t1, NULL);
            if(ret < 0) {
                printf("rknn_run fail! ret=%d\n", ret);
                return -1;
            }

            float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
            avg_time += mytime;
            min_time = std::min(min_time, mytime);
            max_time = std::max(max_time, mytime);
        }
        std::cout << "\nRepeat " << one_pic_repeat_count << " times, avg time per run is " << avg_time / one_pic_repeat_count << " ms\n"<< "max time is " << max_time << " ms, min time is " << min_time << " ms\n";
        std::cout << "--------------------------------------\n";
        // Get Output
        rknn_output outputs[1];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        ret = rknn_outputs_get(ctx, 1, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        // write feature to file
        char format_string[16] = { 0 };
        std::cout << "\n n_output : " << io_num.n_output << "\n";
        
        float *buffer = (float *)outputs[0].buf;
        for (int i = 0; i < 512; i++)
        {
            memset(format_string, 0, 16);
            sprintf(format_string, "%.8f\t", buffer[i]);
            feature_file.write(format_string, strlen(format_string));
        }
        feature_file.write("\n", 1);
        // Release rknn_outputs
        rknn_outputs_release(ctx, 1, outputs);
    } 
    feature_file.close();
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
