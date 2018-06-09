//
//  Fogdetect.m
//  FogDetecting
//
//  Created by J Echoo on 2018/5/22.
//  Copyright © 2018年 J.Echoo. All rights reserved.
//

#import "Fogdetect.h"
#import "UIImage+OpenCV.h"
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<math.h>
#define pi 3.1415926
using namespace std;
using namespace cv;


@implementation Fogdetect

Mat reshape2(Mat &InputMat, int ps)
{
    Mat temp = InputMat.t();
    Mat OutputMat = temp.reshape(0, ps);
    return OutputMat;
}
enum ConvolutionType {
    CONVOLUTION_FULL,
    CONVOLUTION_SAME,
    CONVOLUTION_VALID
};
Mat conv2(const Mat &img, const Mat& ikernel, ConvolutionType type)
{
    Mat dest;
    Mat kernel;
    kernel = ikernel;
    Mat source = img;
    if (CONVOLUTION_FULL == type)
    {
        source = Mat();
        const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
        copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
    }
    cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
    int borderMode = BORDER_CONSTANT;
    filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);
    
    if (CONVOLUTION_VALID == type)
    {
        dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
    }
    return dest;
}

Mat conv22(const Mat &img, const Mat& ikernel, ConvolutionType type)
{
    Mat dest;
    Mat kernel;
    flip(ikernel, kernel, -1);
    Mat source = img;
    if (CONVOLUTION_FULL == type)
    {
        source = Mat();
        const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
        copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
    }
    cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
    int borderMode = BORDER_CONSTANT;
    filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);
    
    if (CONVOLUTION_VALID == type)
    {
        dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
    }
    return dest;
}

//这是特殊的reshape
Mat NewReshape(Mat &InputMat, int ps)
{
    int row = 1;
    int col = InputMat.cols / ps;
    
    Mat *t = new Mat[col];
    Mat *t1 = new Mat[col];
    Mat *xin = new Mat[col];
    for (int i = 0; i < col; i++)
    {
        xin[i] = InputMat(cv::Rect(i*ps, 0, ps, 1));
        t[i] = xin[i].clone();
        t1[i] = t[i].reshape(0, ps);
    }
    Mat outputMat = Mat::zeros(ps, row*col, CV_64F);
    for (int i = 0; i < outputMat.rows; i++)
        for (int j = 0; j < outputMat.cols; j++)
            outputMat.at<double>(i, j) = t1[j].at<double>(i, 0);
    return outputMat;
}

Mat im2col(Mat &InputMat, int ps)
{
    int row = InputMat.rows / ps;
    int col = InputMat.cols / ps;
    Mat **t = new Mat *[row];
    for (int i = 0; i<row; ++i)
        t[i] = new Mat[col];
    
    Mat **t1 = new Mat *[row];
    for (int i = 0; i<row; ++i)
        t1[i] = new Mat[col];
    
    Mat **xin = new Mat *[row];
    for (int i = 0; i<row; ++i)
        xin[i] = new Mat[col];
    
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            
            xin[j][i] = InputMat(cv::Rect(i*ps, j*ps, ps, ps));
            t[j][i] = xin[j][i].clone();
            Mat temp = t[j][i].t();
            t1[j][i] = temp.reshape(0, ps*ps);
            
        }
    
    Mat outputMat = Mat::zeros(ps*ps, row*col, CV_64F);
  
    Mat *t2 = new Mat[row*col];
    int i, j, count = 0;
    for (i = 0; i < col; i++)
    {
        for (j = 0; j < row; j++)
        {
            t2[count] = t1[j][i];
            count++;
        }
    }
    

    for (int i = 0; i < outputMat.rows; i++)
        for (int j = 0; j < outputMat.cols; j++)
            outputMat.at<double>(i, j) = t2[j].at<double>(i, 0);
    return outputMat;
}

Mat var(Mat &inputMat)
{
    Mat outputMat = Mat::zeros(1, inputMat.cols, CV_64F);
    for (int i = 0; i < inputMat.cols; i++)
    {
        double sum = 0, s = 0, e = 0;
        for (int j = 0; j < inputMat.rows; j++)
        {
            sum += inputMat.at<double>(j, i);
        }
        e = sum / inputMat.rows;
        for (int j = 0; j < inputMat.rows; j++)
        {
            s += (inputMat.at<double>(j, i) - e)*(inputMat.at<double>(j, i) - e);
        }
        s = s / (inputMat.rows - 1);
        
        outputMat.at<double>(0, i) = s;
    }
    return outputMat;
}

Mat nanvar(Mat &inputMat)
{
    Mat outputMat = Mat::zeros(1, inputMat.cols, CV_64F);
    for (int i = 0; i < inputMat.cols; i++)
    {
        double sum = 0, s = 0, e = 0;
        int n = inputMat.rows;
        
        for (int j = 0; j < inputMat.rows; j++)
        {
            
            if (inputMat.at<double>(j, i) != 0)
            {
                sum += inputMat.at<double>(j, i);
            }
            else
            {
                n--;
                continue;
            }
        }
        e = sum / n;
        for (int j = 0; j < inputMat.rows; j++)
        {
            if (inputMat.at<double>(j, i) != 0)
            {
                s += (inputMat.at<double>(j, i) - e)*(inputMat.at<double>(j, i) - e);
            }
            else
                continue;
        }
        s = s / (n - 1);
        outputMat.at<double>(0, i) = s;
    }
    return outputMat;
}

Mat circshift(Mat &inputMat, int move)
{
    move = 1;
    Mat outputMat = Mat::zeros(inputMat.rows, inputMat.cols, CV_64F);
    for (int i = 1; i < inputMat.rows; i++)
        for (int j = 0; j < inputMat.cols; j++)
            outputMat.at<double>(i, j) = inputMat.at<double>(i - 1, j);
    for (int i = 0; i < inputMat.cols; i++)
        outputMat.at<double>(0, i) = inputMat.at<double>(inputMat.rows - 1, i);
    return outputMat;
}

Mat NewMean(Mat &inputMat)
{
    Mat outputMat = Mat::zeros(1, inputMat.cols, CV_64F);
    for (int i = 0; i < inputMat.cols; i++)
    {
        double sum = 0, e = 0;
        for (int j = 0; j < inputMat.rows; j++)
        {
            sum += inputMat.at<double>(j, i);
        }
        e = sum / inputMat.rows;
        outputMat.at<double>(0, i) = e;
    }
    return outputMat;
}
//sum
Mat NewSum(Mat &inputMat)
{
    Mat outputMat = Mat::zeros(1, inputMat.cols, CV_64F);
    for (int i = 0; i < inputMat.cols; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputMat.rows; j++)
        {
            sum += inputMat.at<double>(j, i);
        }
        outputMat.at<double>(0, i) = sum;
    }
    return outputMat;
}

//border_in
Mat border_in(Mat &I, int ps)
{
    //Input - input image I, patch size ps
    //Output - border added image
    int uc, dc = 0;
    if (ps % 2 == 0)
    {
        uc = ps / 2;
        dc = ps / 2 - 1;
    }
    else
    {
        uc = ps / 2;
        dc = uc;
    }
    Mat ucb = Mat::zeros(uc, I.cols, CV_64FC1);
    for (int i = 0; i < uc; i++)
        for (int j = 0; j < I.cols; j++)
            ucb.at<double>(i, j) = I.at<double>(i, j);
    Mat dcb = Mat::zeros(dc + 1, I.cols, CV_64FC1);
    for (int i = 0; i < dc + 1; i++)
        for (int j = 0; j < I.cols; j++)
            dcb.at<double>(i, j) = I.at<double>(i + I.rows - dc - 1, j);
    
    Mat Igtemp1 = Mat::zeros(ucb.rows + dcb.rows + I.rows, I.cols, CV_64FC1);
    for (int i = 0; i< ucb.rows; i++)
        for (int j = 0; j<Igtemp1.cols; j++)
            Igtemp1.at<double>(i, j) = ucb.at<double>(i, j);
    for (int i = ucb.rows; i < I.rows + ucb.rows; i++)
        for (int j = 0; j<Igtemp1.cols; j++)
            Igtemp1.at<double>(i, j) = I.at<double>(i - ucb.rows, j);
    for (int i = ucb.rows + I.rows; i < Igtemp1.rows; i++)
        for (int j = 0; j<Igtemp1.cols; j++)
            Igtemp1.at<double>(i, j) = dcb.at<double>(i - ucb.rows - I.rows, j);
    
    Mat lcb = Mat::zeros(Igtemp1.rows, uc, CV_64FC1);
    for (int i = 0; i<Igtemp1.rows; i++)
        for (int j = 0; j<uc; j++)
            lcb.at<double>(i, j) = Igtemp1.at<double>(i, j);
    
    Mat rcb = Mat::zeros(Igtemp1.rows, dc + 1, CV_64FC1);
    for (int i = 0; i<Igtemp1.rows; i++)
        for (int j = 0; j<dc + 1; j++)
            rcb.at<double>(i, j) = Igtemp1.at<double>(i, Igtemp1.cols - dc - 1 + j);
    
    Mat nI = Mat::zeros(lcb.rows, lcb.cols + Igtemp1.cols + rcb.cols, CV_64FC1);
    for (int i = 0; i<lcb.rows; i++)
        for (int j = 0; j<lcb.cols; j++)
            nI.at<double>(i, j) = lcb.at<double>(i, j);
    for (int i = 0; i<lcb.rows; i++)
        for (int j = lcb.cols; j< Igtemp1.cols + lcb.cols; j++)
            nI.at<double>(i, j) = Igtemp1.at<double>(i, j - lcb.cols);
    for (int i = 0; i<lcb.rows; i++)
        for (int j = Igtemp1.cols + lcb.cols; j<nI.cols; j++)
            nI.at<double>(i, j) = rcb.at<double>(i, j - Igtemp1.cols - lcb.cols);
    
    return nI;
}

//border_out
Mat border_out(Mat &I, int ps)
{
    // Input - input image I, patch size ps
    // Output - border trimmed image
    int uc, dc = 0;
    if (ps % 2 == 0)
    {
        uc = ps / 2;
        dc = ps / 2 - 1;
    }
    else
    {
        uc = ps / 2;
        dc = uc;
    }
    Mat temp1 = I(cv::Rect(uc, 0, I.cols - uc, I.rows));
    Mat temp2 = temp1(cv::Rect(0, 0, temp1.cols - dc, temp1.rows));
    Mat temp3 = temp2(cv::Rect(0, uc, temp2.cols, temp2.rows - uc));
    Mat temp4 = temp3(cv::Rect(0, 0, temp3.cols, temp3.rows - dc));
    Mat nI = temp4.clone();
    return nI;
}
//CE 感知对比能对于灰度，蓝黄，红绿颜色通道
void CE(Mat& I, Mat& CE_gray, Mat& CE_by, Mat& CE_rg)
{
    //基本参数
    double sigma = 3.25;
    double semisaturation = 0.1;
    double t1 = 0.2353;
    double t2 = 0.2287;
    double t3 = 0.0528;
    int border_s = 20;
    
    
    int break_off_sigma = 3;
    double filtersize = break_off_sigma*sigma;
    int num = filtersize * 2 + 1;
    Mat x = Mat::zeros(1, num, CV_64F);
    double j = 0;
    for (int i = 0; i < num; i++)
    {
        x.at<double>(0, i) = -filtersize + j;
        j++;
    }
    
    Mat Gauss;
    double temp2 = -2 * sigma * sigma;
    Mat tempx = x.mul(x) / temp2;
    exp(tempx, tempx);
    
    Gauss = 1 / (sqrt(2 * 3.1415926) * sigma)*tempx;
    Scalar Gausstemp = sum(Gauss);
    Gauss = Gauss / Gausstemp.val[0];
    
    Mat Gx = (x.mul(x) / (sigma*sigma*sigma*sigma) - 1 / (sigma*sigma)).mul(Gauss);
    Gx = Gx - sum(Gx).val[0] / x.cols;
    Mat tempmu = 0.5*x.mul(x).mul(Gx);
    Gx = Gx / (sum(tempmu).val[0]);
    
    //颜色转换
    Mat I2;
    I.convertTo(I2, CV_64F);
    vector<Mat> channels;
    Mat B;
    Mat G;
    Mat R;
    split(I2, channels);
    B = channels.at(0);
    G = channels.at(1);
    R = channels.at(2);
    Mat gray = 0.299*R + 0.587*G + 0.114*B;
    Mat by = 0.5*R + 0.5*G - B;
    Mat rg = R - G;
    int row = I2.rows;
    int col = I2.cols;
    int dim = I2.channels();
    CE_gray = Mat::zeros(row, col, CV_64FC1);
    CE_by = Mat::zeros(row, col, CV_64FC1);
    CE_rg = Mat::zeros(row, col, CV_64FC1);
    
    //CE_Gray
    Mat gray_temp1 = border_in(gray, border_s);
    Mat Cx_gray = conv22(gray_temp1, Gx, CONVOLUTION_SAME);
    Mat Cy_gray = conv22(gray_temp1, Gx.t(), CONVOLUTION_SAME);
    Mat C_gray_temp2;
    sqrt((Cx_gray.mul(Cx_gray) + Cy_gray.mul(Cy_gray)), C_gray_temp2);
    Mat C_gray = border_out(C_gray_temp2, border_s);
    Mat Maxcgray; Mat Maxcg;
    reduce(C_gray, Maxcgray, 1, CV_REDUCE_MAX, CV_64F);
    reduce(Maxcgray.t(), Maxcg, 1, CV_REDUCE_MAX, CV_64F);
    double Maxcg1 = Maxcg.at<double>(0, 0);
    Mat R_gray = (C_gray*Maxcg1) / (C_gray + Maxcg1*semisaturation);
    Mat R_gray_temp1 = R_gray - t1;
    for (int i = 0; i < CE_gray.rows; i++)
        for (int j = 0; j < CE_gray.cols; j++)
        {
            if (R_gray_temp1.at<double>(i, j)>0.0000001)
                CE_gray.at<double>(i, j) = R_gray_temp1.at<double>(i, j);
        }
    
    //CE_by
    Mat by_temp1 = border_in(by, border_s);
    Mat Cx_by = conv22(by_temp1, Gx, CONVOLUTION_SAME);
    Mat Cy_by = conv22(by_temp1, Gx.t(), CONVOLUTION_SAME);
    Mat C_by_temp2;
    sqrt((Cx_by.mul(Cx_by) + Cy_by.mul(Cy_by)), C_by_temp2);
    Mat C_by = border_out(C_by_temp2, border_s);
    Mat Maxcby; Mat Maxcbyy;
    reduce(C_by, Maxcby, 1, CV_REDUCE_MAX, CV_64F);
    reduce(Maxcby.t(), Maxcbyy, 1, CV_REDUCE_MAX, CV_64F);
    double Maxcbyy1 = Maxcbyy.at<double>(0, 0);
    Mat  R_by = (C_by*Maxcbyy1) / (C_by + Maxcbyy1*semisaturation);
    Mat R_by_temp1 = R_by - t2;
    for (int i = 0; i < CE_by.rows; i++)
        for (int j = 0; j < CE_by.cols; j++)
        {
            if (R_by_temp1.at<double>(i, j)>0.0000001)
                CE_by.at<double>(i, j) = R_by_temp1.at<double>(i, j);
        }
    //CE_rg
    Mat rg_temp1 = border_in(rg, border_s);
    Mat Cx_rg = conv22(rg_temp1, Gx, CONVOLUTION_SAME);
    Mat Cy_rg = conv22(rg_temp1, Gx.t(), CONVOLUTION_SAME);
    Mat C_rg_temp2;
    sqrt((Cx_rg.mul(Cx_rg) + Cy_rg.mul(Cy_rg)), C_rg_temp2);
    Mat C_rg = border_out(C_rg_temp2, border_s);
    Mat Maxcrg; Mat Maxcrgg;
    reduce(C_rg, Maxcrg, 1, CV_REDUCE_MAX, CV_64F);
    reduce(Maxcrg.t(), Maxcrgg, 1, CV_REDUCE_MAX, CV_64F);
    double Maxcrgg1 = Maxcrgg.at<double>(0, 0);
    Mat  R_rg = (C_rg*Maxcrgg1) / (C_rg + Maxcrgg1*semisaturation);
    Mat R_rg_temp1 = R_rg - t3;
    for (int i = 0; i < CE_rg.rows; i++)
        for (int j = 0; j < CE_rg.cols; j++)
        {
            if (R_rg_temp1.at<double>(i, j)>0.0000001)
                CE_rg.at<double>(i, j) = R_rg_temp1.at<double>(i, j);
        }
}

//num2cell  num2cell(Mat a,1);
Mat* num2cell(Mat &inputMat)
{
    Mat *a = new Mat[inputMat.cols];
    for (int i = 0; i < inputMat.cols; i++)
        a[i] = inputMat(cv::Rect(i, 0, 1, inputMat.rows));
    return a;
}

//num2cell(Mat a,2);
Mat* num2cell2(Mat &inputMat)
{
    Mat *a = new Mat[inputMat.rows];
    for (int i = 0; i < inputMat.rows; i++)
        a[i] = inputMat(cv::Rect(0, i, inputMat.cols, 1));
    return a;
}

//sqrt
Mat Newsqrt(Mat &inputMat)
{
    Mat outputMat;
    sqrt(inputMat, outputMat);
    return outputMat;
}

//repmat
Mat repmat(Mat &inputMat, int row, int col)
{
    Mat outputMat = Mat::zeros(inputMat.rows*row, inputMat.cols*col, CV_64F);
    for (int i = 0; i < outputMat.rows; i++)
        for (int j = 0; j < outputMat.cols; j++)
        {
            int k = i%inputMat.rows;
            int m = j%inputMat.cols;
            outputMat.at<double>(i, j) = inputMat.at<double>(k, m);
        }
    return outputMat;
}

//cunsum 默认，按照列累加
Mat cumsum(Mat &inputMat)
{
    Mat outputMat = inputMat.clone();
    for (int i = 0; i < inputMat.cols; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputMat.rows; j++)
        {
            sum += inputMat.at<double>(j, i);
            outputMat.at<double>(j, i) = sum;
        }
    }
    return outputMat;
}
//按照行累加
Mat cumsum2(Mat &inputMat)
{
    Mat outputMat = inputMat.clone();
    for (int i = 0; i < inputMat.rows; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputMat.cols; j++)
        {
            sum += inputMat.at<double>(i, j);
            outputMat.at<double>(i, j) = sum;
        }
    }
    return outputMat;
}
//图像熵
double Entropy(Mat img)
{
    
    double temp[256];
    for (int i = 0; i<256; i++)
    {
        temp[i] = 0.0;
    }
    
    // 计算每个像素的累积值
    for (int m = 0; m<img.rows; m++)
    {
        double* t = img.ptr<double>(m);
        for (int n = 0; n<img.cols; n++)
        {
            int i = t[n];
            temp[i] = temp[i] + 1;
        }
    }
    // 计算每个像素的概率
    for (int i = 0; i<256; i++)
    {
        temp[i] = temp[i] / (img.rows*img.cols);
    }
    double result = 0;
    // 根据定义计算图像熵
    for (int i = 0; i<256; i++)
    {
        if (temp[i] == 0.0)
            result = result;
        else
            result = result - temp[i] * (log(temp[i]) / log(2.0));
    }
    return result;
}


+ (CGFloat)Fogdetecting:(UIImage *)image{
    cv::Mat inputMat;
    cv::Mat I;//outputMat
    cv::Mat tmp;
    
    inputMat = [image cvMatImage];
    // Mat I = imread(image);
    // 压缩
    cv::resize(inputMat, tmp, cv::Size(inputMat.rows / 2, inputMat.cols/ 2));
    I = tmp;
    
    int ps = 8;
    int row = I.rows;
    int col = I.cols;
    int dim = I.channels();
    
    int patch_row_num = row / ps;
    int patch_col_num = col / ps;
    //ROI处理，为了使整个图像能够被ps整除
    Mat I3 = I(cv::Rect(0, 0, patch_col_num * ps, patch_row_num * ps));
    Mat I2 = I3.clone();
    row = I2.rows;
    col = I2.cols;
    dim = I2.channels();
    
    //RGB和灰度通道提取
    vector<Mat> channels;
    Mat Ig;
    Mat Blue;
    Mat Green;
    Mat Red;
    cvtColor(I2, Ig, CV_BGR2GRAY);
    split(I2, channels);
    Blue = channels.at(0);
    Green = channels.at(1);
    Red = channels.at(2);
    //暗通道先验
    Mat blue = Mat_<double>(Blue);
    Mat green = Mat_<double>(Green);
    Mat red = Mat_<double>(Red);
    Mat Irn = red / 255;
    Mat Ign = green / 255;
    Mat Ibn = blue / 255;
    Mat Id;
    min(Irn, Ign, Id);
    min(Id, Ibn, Id);
    
    // HSV颜色空间的Is：饱和度
    Mat I_hsv;
    Mat Is;
    vector<Mat> channels2;
    cvtColor(I2, I_hsv, CV_BGR2HSV);
    split(I_hsv, channels2);
    Is = channels2.at(1);
    Is.convertTo(Is, CV_64F);
    Is = Is / 255;
    
    //  MSCN
    Mat MSCN_window = (Mat_<double>(7, 7) << 1.577586358082699e-04, 9.900950389243662e-04, 0.002980486298974, 0.004303520521061, 0.00298048629897400, 0.000990095038924366, 0.000157758635808270, 0.000990095038924366, 0.00621384801586408, 0.0187055667861054, 0.0270089449999423, 0.0187055667861054, 0.00621384801586408, 0.000990095038924366, 0.00298048629897427, 0.0187055667861054, 0.0563094282151982, 0.0813051145166149, 0.0563094282151982, 0.0187055667861054, 0.00298048629897427, 0.00430352052106079, 0.0270089449999423, 0.0813051145166149, 0.117396355390013, 0.0813051145166149, 0.0270089449999423, 0.00430352052106079, 0.00298048629897427, 0.0187055667861054, 0.0563094282151982, 0.0813051145166149, 0.0563094282151982, 0.0187055667861054, 0.00298048629897427, 0.000990095038924366, 0.00621384801586408, 0.0187055667861054, 0.0270089449999423, 0.0187055667861054, 0.00621384801586408, 0.000990095038924366, 0.000157758635808270, 0.000990095038924366, 0.00298048629897427, 0.00430352052106079, 0.00298048629897427, 0.000990095038924366, 0.000157758635808270);
    //格式转换 uchar 到double
    Mat Ignew;
    Ig.convertTo(Ignew, CV_64F);
    Mat mu = Mat::zeros(Ig.rows, Ig.cols, CV_64FC1);
    mu = conv2(Ignew, MSCN_window, CONVOLUTION_SAME);
    Mat DIg = Ignew.mul(Ignew);
    Mat Mattemp = conv2(DIg, MSCN_window, CONVOLUTION_SAME);
    Mat sigma;
    Mat Mattemp1;
    Mat mu_sq = Mat::zeros(mu.rows, mu.cols, CV_64FC1);
    mu_sq = mu.mul(mu);
    absdiff(Mattemp, mu_sq, Mattemp1);
    sqrt(Mattemp1, sigma);
    Mat temp1 = Ignew - mu;
    Mat temp2 = sigma + 1;
    Mat MSCN;
    MSCN = temp1 / temp2;
    Mat cv = sigma / mu;
    // rg and by 通道
    Mat R, G, B;
    Red.convertTo(R, CV_64F);
    Green.convertTo(G, CV_64F);
    Blue.convertTo(B, CV_64F);
    Mat rg = R - G;
    Mat by = 0.5*(R + G) - B;
    
    // 雾感统计特征抽取
    
    //f1
    Mat tempbb = im2col(MSCN, ps);
    Mat tempcc = var(tempbb);//
    Mat MSCN_var = NewReshape(tempcc, row / ps);
    
    //f2,f3
    Mat temp_vertical = circshift(MSCN, 1);
    Mat temp_vertical2 = MSCN.mul(temp_vertical);
    Mat MSCN_V_pair_col = im2col(temp_vertical2, ps);
    
    
    Mat MSCN_V_pair_col_temp1 = MSCN_V_pair_col.clone();
    for (int i = 0; i < MSCN_V_pair_col_temp1.rows; i++)
        for (int j = 0; j < MSCN_V_pair_col_temp1.cols; j++)
            if (MSCN_V_pair_col_temp1.at<double>(i, j)>0)
                MSCN_V_pair_col_temp1.at<double>(i, j) = 0;
    Mat MSCN_V_pair_col_temp2 = MSCN_V_pair_col.clone();
    for (int i = 0; i < MSCN_V_pair_col_temp2.rows; i++)
        for (int j = 0; j < MSCN_V_pair_col_temp2.cols; j++)
            if (MSCN_V_pair_col_temp2.at<double>(i, j)<0)
                MSCN_V_pair_col_temp2.at<double>(i, j) = 0;
    
   // Mat MSCN_V_pair_L_var = NewReshape(nanvar(MSCN_V_pair_col_temp1), row / ps);
    Mat MSCN_V_pair_L_var1=nanvar(MSCN_V_pair_col_temp1);
    Mat MSCN_V_pair_L_var=NewReshape(MSCN_V_pair_L_var1,row/ps);
    
   // Mat MSCN_V_pair_R_var = NewReshape(nanvar(MSCN_V_pair_col_temp2), row / ps);
    Mat MSCN_V_pair_R_var1=nanvar(MSCN_V_pair_col_temp2);
    Mat MSCN_V_pair_R_var= NewReshape(MSCN_V_pair_R_var1,row/ps);
    
    //f4
   // Mat Mean_sigma = NewReshape(NewMean(im2col(sigma, ps)), row / ps);
    Mat Mean_sigma1=im2col(sigma, ps);
    Mat Mean_sigma2=NewMean(Mean_sigma1);
    Mat Mean_sigma=NewReshape(Mean_sigma2, row/ps);
    
    //f5
    //Mat Mean_cv = NewReshape(NewMean(im2col(cv, ps)), row / ps);
    Mat Mean_cv1=im2col(cv, ps);
    Mat Mean_cv2=NewMean(Mean_cv1);
    Mat Mean_cv =NewReshape(Mean_cv2,row/ps);
    
    //f6, f7, f8
    Mat CE_gray;
    Mat CE_by;
    Mat CE_rg;
    CE(I2, CE_gray, CE_by, CE_rg);
   // Mat Mean_CE_gray = NewReshape(NewMean(im2col(CE_gray, ps)), row / ps);
    Mat Mean_CE_gray1=im2col(CE_gray, ps);
    Mat Mean_CE_gray2=NewMean(Mean_CE_gray1);
    Mat Mean_CE_gray= NewReshape(Mean_CE_gray2,row/ps);
    
    
   //Mat Mean_CE_by = NewReshape(NewMean(im2col(CE_by, ps)), row / ps);
    Mat Mean_CE_by1=im2col(CE_by, ps);
    Mat Mean_CE_by2=NewMean(Mean_CE_by1);
    Mat Mean_CE_by= NewReshape(Mean_CE_by2,row/ps);
    
    
    
  //  Mat Mean_CE_rg = NewReshape(NewMean(im2col(CE_rg, ps)), row / ps);
    
    Mat Mean_CE_rg1=im2col(CE_rg, ps);
    Mat Mean_CE_rg2=NewMean(Mean_CE_rg1);
    Mat Mean_CE_rg= NewReshape(Mean_CE_rg2,row/ps);
    
    //f9
    Mat uintIg;
    Ig.convertTo(uintIg, CV_64F);
    Mat tempuIg = im2col(uintIg, ps);
    Mat *IE_temp = num2cell(tempuIg);
    Mat entropymat = Mat::zeros(1, uintIg.cols*uintIg.rows / ps / ps, CV_64FC1);
    for (int i = 0; i < entropymat.cols; i++)
    {
        double e = Entropy(IE_temp[i]);
        entropymat.at<double>(0, i) = e;
    }
    Mat IE = NewReshape(entropymat, row / ps);
    //f10
    //Mat Mean_Id = NewReshape( NewMean(im2col(Id, ps)), row / ps);
    Mat t1=im2col(Id, ps);
    Mat t2=NewMean(t1);
    Mat Mean_Id=NewReshape(t2, row/ps);
    
    //f11
    Mat Isnew=Is.clone();
   // Mat Mean_Is = NewReshape( NewMean(im2col(Isnew, ps)), row / ps);
    Mat Mean_Is1=im2col(Isnew, ps);
    Mat Mean_Is2=NewMean(Mean_Is1);
    Mat Mean_Is=NewReshape(Mean_Is2,row/ps);
    
    
    //f12
    //Mat varrg = var(im2col(rg, ps));
    Mat varrg1=im2col(rg, ps);
    Mat varrg= var(varrg1);
    
   // Mat varby = var(im2col(by, ps));
    Mat varby1=im2col(by, ps);
    Mat varby=var(varrg1);
    
    Mat Sumrgby = varrg + varby;
    Mat sqrtrgby1 = Newsqrt(Sumrgby);
    Mat im2colrg = im2col(rg, ps);
    Mat Meanrg = NewMean(im2colrg);
    Mat Meanrgf = Meanrg.mul(Meanrg);
    Mat im2colby = im2col(by, ps);
    Mat Meanby = NewMean(im2colby);
    Mat Meanbyf = Meanby.mul(Meanby);
    Mat Sumrgby2 = Meanrgf + Meanbyf;
    Mat sqrtrgby2 = 0.3* Newsqrt(Sumrgby2);
    Mat sumlast = sqrtrgby1 + sqrtrgby2;
    Mat CF = NewReshape(sumlast, row / ps);
    Mat feat = Mat::zeros((MSCN_var.rows*MSCN_var.cols), 12, CV_64FC1);
    Mat MSCN_var_new = reshape2(MSCN_var, feat.rows);
    Mat MSCN_V_pair_R_var_new = reshape2(MSCN_V_pair_R_var, feat.rows);
    Mat MSCN_V_pair_L_var_new = reshape2(MSCN_V_pair_L_var, feat.rows);
    Mat Mean_sigma_new = reshape2(Mean_sigma, feat.rows);
    Mat Mean_cv_new = reshape2(Mean_cv, feat.rows);
    Mat Mean_CE_gray_new = reshape2(Mean_CE_gray, feat.rows);
    Mat Mean_CE_by_new = reshape2(Mean_CE_by, feat.rows);
    Mat Mean_CE_rg_new = reshape2(Mean_CE_rg, feat.rows);
    Mat IE_new = reshape2(IE, feat.rows);
    Mat Mean_Id_new = reshape2(Mean_Id, feat.rows);
    Mat Mean_Is_new = reshape2(Mean_Is, feat.rows);
    Mat CF_new = reshape2(CF, feat.rows);
    for (int j = 0; j < feat.rows; j++)
    {
        feat.at<double>(j, 0) = MSCN_var_new.at<double>(j, 0);
        feat.at<double>(j, 1) = MSCN_V_pair_R_var_new.at<double>(j, 0);
        feat.at<double>(j, 2) = MSCN_V_pair_L_var_new.at<double>(j, 0);
        feat.at<double>(j, 3) = Mean_sigma_new.at<double>(j, 0);
        feat.at<double>(j, 4) = Mean_cv_new.at<double>(j, 0);
        feat.at<double>(j, 5) = Mean_CE_gray_new.at<double>(j, 0);
        feat.at<double>(j, 6) = Mean_CE_by_new.at<double>(j, 0);
        feat.at<double>(j, 7) = Mean_CE_rg_new.at<double>(j, 0);
        feat.at<double>(j, 8) = IE_new.at<double>(j, 0);
        feat.at<double>(j, 9) = Mean_Id_new.at<double>(j, 0);
        feat.at<double>(j, 10) = Mean_Is_new.at<double>(j, 0);
        feat.at<double>(j, 11) = CF_new.at<double>(j, 0);
    }
    log((1 + feat), feat);
    
    //MVG model distance
        //载入自然无雾图像库的特征(mu, cov)
    Mat cov_fogfreeparam = (Mat_<double>(12, 12) << 0.00413222407411700, 0.00166822077977077, 0.00174884851477932, 0.00328856303949822, -0.000338939833962396, -0.00246654408328581, -0.00393616524728224, -0.00415024883038623, 0.000666897363540972, 0.000481756999640222, -0.000132961571250451, -0.00242595074316808, 0.00166822077977077, 0.00351948710009588, 2.05671786785027e-05, 0.00112048362544143, -0.000356571451371784, -0.00135106206389485, -0.000971889732820689, -0.000353756788563541, 0.000210655714351528, 0.000207419089504934, 2.74738705119173e-05, -0.000193224656461530, 0.00174884851477932, 2.05671786785027e-05, 0.00270380870633144, 0.00119709534377084, 0.000126414035681717, -0.000810939394384069, -0.00103795020814272, -0.00178332306727464, 0.000113376198996303, -5.16431799689723e-05, 0.000110130983798381, -0.000730444324567762, 0.00328856303949822, 0.00112048362544143, 0.00119709534377084, 0.109883582519708, 0.0196122053059988, 0.0371979396577043, 0.0317952012566110, 0.0246238134420843, 0.00753778682456658, 0.00629079183028243, -0.000165908044198588, 0.0435758249092768, -0.000338939833962396, -0.000356571451371784, 0.000126414035681717, 0.0196122053059988, 0.0141317744100117, 0.00816482000325287, 0.00284914934469880, 0.000970095961576604, 0.000114574464372470, -0.00284895705881976, -0.000226974411477168, -0.000226974411477168, -0.00246654408328581, -0.00135106206389485, -0.000810939394384069, 0.0371979396577043, 0.00816482000325287, 0.0310538715450125, 0.0302512677814476, 0.0248575090871819, 0.00229834098340289, 0.00103028576421454, 0.000889113968990091, 0.0254625115005395, -0.00393616524728224, -0.000971889732820689, -0.00103795020814272, 0.0317952012566110, 0.00284914934469880, 0.0302512677814476, 0.137325802727451, 0.0913584358139026, 0.00182165606582398, -0.00348998272593340, 0.0123088688771617, 0.130196167528451, -0.00415024883038623, -0.000353756788563541, -0.00178332306727464, 0.0246238134420843, 0.000970095961576604, 0.0248575090871819, 0.0913584358139026, 0.139595794349651, 0.00115647060286666, 0.000147675289191342, 0.00817855040181794, 0.126346723359608, 0.000666897363540972, 0.000210655714351528, 0.000113376198996303, 0.00753778682456658, 0.000114574464372470, 0.00229834098340289, 0.00182165606582398, 0.00115647060286666, 0.00153657180928221, 0.000846535394861678, -0.000117796590288973, 0.00429136311820265, 0.000481756999640222, 0.000207419089504934, -5.16431799689723e-05, 0.00629079183028243, -0.00284895705881976, 0.00103028576421454, -0.00348998272593340, 0.000147675289191342, 0.000846535394861678, 0.00423914967217232, -0.00340355843851411, 0.00206699408710125, -0.000132961571250451, 2.74738705119173e-05, 0.000110130983798381, -0.000165908044198588, -0.000226974411477168, 0.000889113968990091, 0.0123088688771617, 0.00817855040181794, -0.000117796590288973, -0.00340355843851411, 0.00735205864005300, 0.0234249200577809, -0.00242595074316808, -0.000193224656461530, -0.000730444324567762, 0.0435758249092768, -0.0141581226090149, 0.0254625115005395, 0.130196167528451, 0.126346723359608, 0.00429136311820265, 0.00206699408710125, 0.0234249200577809, 0.248287080488324);
    Mat mu_fogfreeparam = (Mat_<double>(1, 12) << 0.389774224937378, 0.119555520885610, 0.0738060416516676, 3.21585250666259, 0.292762343523965, 1.86225854558008, 1.15028033475630, 1.03156456057672, 1.84138722167384, 0.171471878267803, 0.443016994045952, 3.30373193302860);
    // 对于每一个块测试参数
    Mat mu_fog_param_patch = feat.clone();
    Mat tfeat = feat.t();
    Mat cov_fog_param_patch = (nanvar(tfeat)).t();
    // 距离计算
    int feature_size = feat.cols;
    Mat mu_matrix = repmat(mu_fogfreeparam, feat.rows, 1) - mu_fog_param_patch;
    Mat tempbd1 = Mat::ones(1, cov_fog_param_patch.rows, CV_64F);
    tempbd1 = feature_size*tempbd1;
    Mat tempbd2 = cumsum2(tempbd1);
    Mat cov_temp1 = Mat::zeros(1, feature_size*tempbd2.cols, CV_64F);
    for (int j = 0; j < tempbd2.cols; j++)
    {
        int tempp = tempbd2.at<double>(0, j);
        cov_temp1.at<double>(0, tempp - 1) = 1;
    }
    Mat tempbd3 = cumsum2(cov_temp1) - cov_temp1 + 1;
    Mat cov_temp2 = Mat::zeros(tempbd3.cols, 1, CV_64F);;
    for (int i = 0; i < cov_temp2.rows; i++)
        for (int j = 0; j < cov_temp2.cols; j++)
            cov_temp2.at<double>(i, j) = cov_fog_param_patch.at<double>((tempbd3.at<double>(j, i) - 1), 0);
    Mat cov_temp3 = repmat(cov_temp2, 1, feature_size);
    Mat cov_temp4 = repmat(cov_fogfreeparam, cov_fog_param_patch.rows, 1);
    Mat cov_matrix = (cov_temp3 + cov_temp4) / 2;
    
    // 孢元计算
    Mat *mu_cell = num2cell2(mu_matrix);
    //mat2cell
    Mat tempbd4 = Mat::ones(1, mu_matrix.rows, CV_64F);
    tempbd4 = feature_size*tempbd4;
    Mat **cov_cell = new Mat *[mu_matrix.rows];
    for (int i = 0; i<mu_matrix.rows; ++i)
        cov_cell[i] = new Mat[1];
    
    for (int i = 0; i < mu_matrix.rows; i++)
        for (int j = 0; j < 1; j++)
        {
            Mat tempcov_cell = cov_matrix(cv::Rect(0, feature_size*i, feature_size, feature_size));
            cov_cell[i][j] = tempcov_cell.clone();
        }
    Mat mu_matrix_t = mu_matrix.t();
    
    Mat *mu_transpose_cell = num2cell(mu_matrix_t);
    
   // 雾等级计算
    Mat** distance_patch_t1 = new Mat *[mu_matrix.rows];
    for (int i = 0; i<mu_matrix.rows; ++i)
        distance_patch_t1[i] = new Mat[1];
    for (int i = 0; i < mu_matrix.rows; i++)
        for (int j = 0; j < 1; j++)
            distance_patch_t1[i][j] = mu_cell[i] * (cov_cell[i][j].inv());
    Mat** distance_patch_t2 = new Mat *[mu_matrix.rows];
    for (int i = 0; i<mu_matrix.rows; ++i)
        distance_patch_t2[i] = new Mat[1];
    for (int i = 0; i < mu_matrix.rows; i++)
        for (int j = 0; j < 1; j++)
            distance_patch_t2[i][j] = distance_patch_t1[i][j] * (mu_transpose_cell[i]);
    
    Mat distance_patch = Mat::zeros(distance_patch_t2[0][0].rows*mu_matrix.rows, distance_patch_t2[0][0].cols, CV_64F);
    for (int i = 0; i < distance_patch.rows; i++)
        for (int j = 0; j < distance_patch.cols; j++)
        {
            distance_patch.at<double>(i, j) = distance_patch_t2[i][0].at<double>(0, j);
        }
    
    sqrt(distance_patch, distance_patch);
    Mat Df = NewMean(distance_patch);
    Mat distance_patch_t = distance_patch.t();
    Mat Df_map = NewReshape(distance_patch_t, row / ps);
    
    //Dff
    //载入自然雾霾图像库的特征 (mu, cov)
    Mat cov_foggyparam = (Mat_<double>(12, 12) << 0.00425493967462267, 0.000687307694274529, 0.000467958997731953, 0.0145723846032284, 0.000171536146737019, 0.00500750485403297, 0.00297949369225151, 0.00362352450977183, 0.0150184944589597, -0.00130751808591846, 0.000195742608784787, 0.00843573677427318, 0.000687307694274529, 0.000189892692311164, 7.34839048307997e-05, 0.00258890813369798, 3.38790045034198e-05, 0.00104454798525700, 0.000525926716245123, 0.000600145222537705, 0.00229510521120643, -0.000255223012065669, 3.30126312347205e-05, 0.00129409309458668, 0.000467958997731953, 7.34839048307997e-05, 0.000137219020714782, 0.00166151656687392, 2.06179222895593e-05, 0.000555286942953741, 0.000364962639783096, 0.000419026478785233, 0.00143686449576838, -0.000174301738437431, 2.32308158091524e-05, 0.00101346205816686, 0.0145723846032284, 0.00258890813369798, 0.00166151656687392, 0.0807248415599781, 0.00107793965748349, 0.0413238995231783, 0.0154418613143800, 0.0170438798534783, 0.0763714069045161, -0.00919162625748854, 0.000948157883593691, 0.0337809694697541, 0.000171536146737019, 3.38790045034198e-05, 2.06179222895593e-05, 0.00107793965748349, 1.84695984160031e-05, 0.000583793058564611, 0.000201571675005027, 0.000213156393841568, 0.000889775217897222, -0.000158005976541900, 1.07208232511606e-05, 0.000336911331937395, 0.00500750485403297, 0.00104454798525700, 0.000555286942953741, 0.0413238995231783, 0.000583793058564611, 0.0441181026191870, 0.0130265399484672, 0.0123205660837335, 0.0388399785620150, -0.00455846704948788, 0.000335151974741357, 0.0139394484586678, 0.00297949369225151, 0.000525926716245123, 0.000364962639783096, 0.0154418613143800, 0.000201571675005027, 0.0130265399484672, 0.0257542409152021, 0.0180084602810093, 0.0189034770075237, -0.00166040013440690, 0.000686355002117748, 0.0381882717810330, 0.00362352450977183, 0.000600145222537705, 0.000419026478785233, 0.0170438798534783, 0.000213156393841568, 0.0123205660837335, 0.0180084602810093, 0.0392119574443634, 0.0217489507929472, -0.00261456139426273, 0.00174426958269103, 0.0640158755244730, 0.0150184944589597, 0.00229510521120643, 0.00143686449576838, 0.0763714069045161, 0.000889775217897222, 0.0388399785620150, 0.0189034770075237, 0.0217489507929472, 0.116664940562771, -0.00835839319600369, 0.00135528400219975, 0.0478174518490932, -0.00130751808591846, -0.000255223012065669, -0.000174301738437431, -0.00919162625748854, -0.000158005976541900, -0.00455846704948788, -0.00166040013440690, -0.00261456139426273, -0.00835839319600369, 0.00466774077674801, -0.000662238247222017, -0.00867906645248512, 0.000195742608784787, 3.30126312347205e-05, 2.32308158091524e-05, 0.000948157883593691, 1.07208232511606e-05, 0.000335151974741357, 0.000686355002117748, 0.00174426958269103, 0.00135528400219975, -0.000662238247222017, 0.000994075899746345, 0.0154898036288386, 0.00843573677427318, 0.00129409309458668, 0.00101346205816686, 0.0337809694697541, 0.000336911331937395, 0.0139394484586678, 0.0381882717810330, 0.0640158755244730, 0.0478174518490932, -0.00867906645248512, 0.0154898036288386, 0.342746484807520);
    Mat mu_foggyparam = (Mat_<double>(1, 12) << 0.0999661533376065, 0.0105990963207700, 0.00829623054210693, 0.608056155521870, 0.00480860914377503, 0.187966356680982, 0.131318836474065, 0.236831591971413, 0.912616181510066, 0.568023541658438, 0.0418215709763005, 1.31548016737338);
    //距离计算
    Mat mu_matrix1 = repmat(mu_foggyparam, feat.rows, 1) - mu_fog_param_patch;
    Mat cov_temp5 = repmat(cov_foggyparam, cov_fog_param_patch.rows, 1);
    Mat cov_matrix1 = (cov_temp3 + cov_temp5) / 2;
    
    //孢元计算
    Mat *mu_cell1 = num2cell2(mu_matrix1);
    
    //mat2cell
    Mat tempbd41 = Mat::ones(1, mu_matrix1.rows, CV_64F);
    tempbd41 = feature_size*tempbd41;
    Mat **cov_cell1 = new Mat *[mu_matrix1.rows];
    for (int i = 0; i<mu_matrix1.rows; ++i)
        cov_cell1[i] = new Mat[1];
    
    for (int i = 0; i < mu_matrix1.rows; i++)
        for (int j = 0; j < 1; j++)
        {
            Mat tempcov_cell1 = cov_matrix1(cv::Rect(0, i*feature_size, feature_size, feature_size));
            cov_cell1[i][j] = tempcov_cell1.clone();
        }
    Mat mu_matrix_t1 = mu_matrix1.t();
    Mat *mu_transpose_cell1 = num2cell(mu_matrix_t1);
    
    Mat** distance_patch_t11 = new Mat *[mu_matrix1.rows];
    for (int i = 0; i<mu_matrix1.rows; ++i)
        distance_patch_t11[i] = new Mat[1];
    for (int i = 0; i < mu_matrix1.rows; i++)
        for (int j = 0; j < 1; j++)
            distance_patch_t11[i][j] = mu_cell1[i] * (cov_cell1[i][j].inv());
    
    Mat** distance_patch_t21 = new Mat *[mu_matrix1.rows];
    for (int i = 0; i<mu_matrix1.rows; ++i)
        distance_patch_t21[i] = new Mat[1];
    for (int i = 0; i < mu_matrix1.rows; i++)
        for (int j = 0; j < 1; j++)
            distance_patch_t21[i][j] = distance_patch_t11[i][j] * (mu_transpose_cell1[i]);
    
    Mat distance_patch1 = Mat::zeros(distance_patch_t21[0][0].rows*mu_matrix1.rows, distance_patch_t21[0][0].cols, CV_64F);
    for (int i = 0; i < distance_patch1.rows; i++)
        for (int j = 0; j < distance_patch1.cols; j++)
        {
            distance_patch1.at<double>(i, j) = distance_patch_t21[i][0].at<double>(0, j);
        }
    sqrt(distance_patch1, distance_patch1);
    Mat Dff = NewMean(distance_patch1);
    Mat distance_patch1_t = distance_patch1.t();
    Mat Dff_map = NewReshape(distance_patch1_t, row / ps);
    
    // 感知雾密度和雾密度图
    Mat D = Df / (Dff + 1);
    Mat D_map = Df_map / (Dff_map + 1);
    CGFloat fogvalue = D.at<double>(0, 0);
//    if (fogvalue > 3)
//        level = "∏ﬂ";
//    else if (fogvalue > 1 && fogvalue <= 3)
//        level = "÷–";
//    else
//        level = "µÕ";
//    UpdateData(false);
    //imshow("‘≠ ºÕº", I);
    //waitKey(0);
    return fogvalue;
}




@end
