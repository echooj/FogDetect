![zhang-kaiyv-FmyIBz2JDHU-unsplash3.jpg](https://upload-images.jianshu.io/upload_images/10826585-b45aa84084a710b3.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




# 一. 前言：

当年在武大电信院做毕业设计的时候，我的导师希望我用C代码完成彩色图像的雾霾程度评价，如果时间允许，能够开发出一款APP完成移动端的实现。很可惜，当时我对于iOS开发一无所知，时间不允许我完成移动端的实现，但是完成了Windows端的实现，利用MFC做了一个毕业设计展示。还记得当时花了很多时间研究图像处理，一步一步实现每一个算法细节，辛苦却可以感受到满满的收获。

花了很长时间利用下班时间自学iOS,技术还是很渣，希望大家能够勉强看得下去我的文章。

图像中雾霾程度的评价是图像增强处理的第一步，从方法上可以分为主观评价方法和客观评价方法，前者凭感知者主观感受评价;后者依据模型给出的量化指标或参数衡量。客观评价的目标使得评价模型准确地反映人眼视觉感知的主观评价。本课题研究基于自然场景统计特征(NSS)和其他感知特征的雾霾浓度评价研究，使得对于输入的图像，准确输出评价值。

先给展示一下demo
![demo.gif](https://upload-images.jianshu.io/upload_images/10826585-494a97a3a5182bc5.gif?imageMogr2/auto-orient/strip)
操作步骤：
1.选取图片，然后可以从手机相册中选择一张图片，也可以拍照
2.可以修改尺寸大小，点击完成
3.点击雾霾程度分析的button，就可以看到给出这张图片的雾霾程度的客观评价值

#二. 前期准备
考虑到 OpenCV 是基于 C/C++ 可跨平台的通用 Lib，为了降低学习成本，便将整个学习和实验集成到 iOS 的开发环境里了。前期要做如下几方面的准备工作：

1.  直接到OpenCV官网下载你想要的OpenCV的framework；
2.  将 OpenCV.framework 导入 iOS 项目工程中；
3.  因为 OpenCV 中的 MIN 宏和 UIKit 的 MIN 宏有冲突，所以需要在 .pch 文件中，先定义 OpenCV 的头文件，否则会有编译错误；
4.  将需要混编 C++ 和 Objective-C 的文件后缀改为 **.mm**;
5.  为 UIImage 添加 Category，方便与OpenCV 图象格式的数据 cv::Mat 相互转换。
    因这些繁琐的配置问题不是本文写作重点，而且网上不乏一些详细说明，推荐参考唐巧大神的 [在MacOS和iOS系统中使用OpenCV](https://link.jianshu.com?t=https%3A%2F%2Fblog.devtang.com%2F2012%2F10%2F27%2Fuse-opencv-in-ios%2F) 一文，这里就不再赘述。

# 三.  算法：基于自然场景统计特征（NSS）和其他的雾感统计特征的雾霾程度分析

下面是算法的实现流程图：

![image](http://upload-images.jianshu.io/upload_images/10826585-bfee48a8b7d7610f?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

简单介绍下个这个算法：

空域 NSS 模型即计算自然图像上的局部差值均值和对比度归一化(MSCN) 系数。雾感统计特征包括 MSCN 系数，局部平均和局部系数的锐利度方差，对比能，图像熵，像素暗通道先验，色彩饱和度和色彩丰富度等，总共 12个雾感知统计特征，利用这些雾感统计特征去计算每一个已经被分割为 P×P 大小 的图像块，然后得到图像的多元高斯(MVG)模型，利用 Mahalanobis 类距离测量方法，测量图像的雾感知统计特征的 MVG 模型分别与自然雾霾图像库的 MVG模型和自然无雾图像库的 MVG 模型之间的距离，最后计算得到感知雾霾程度。

# 四.算法具体实现细节和主要代码

这里仅贴一下部分我认为比较重要的代码，具体实现细节请参考我的github：[https://github.com/echooj/FogDetect](https://github.com/echooj/FogDetect)
当然，懒得看代码的同学可以直接跳到测试结果部分，就是有一张可爱的小蘑菇图的位置
###基础过程
1. 图像尺寸的测量和 RGB 值的获取

先提取图像的尺寸，然后修改图像尺寸，使得图像的宽和高能够被8整除，最后提取修改后的图像的R、G、B值和灰度值Ig。
```
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
```
2. HSV 空间的转换
将图像转换到HSV空间，并获取s(饱和度)的值。
```
Mat I_hsv;
Mat Is;
vector<Mat> channels2;
cvtColor(I2, I_hsv, CV_BGR2HSV);
split(I_hsv, channels2);
Is = channels2.at(1);
Is.convertTo(Is,CV_64F);
Is=Is/255;
```
3. 暗通道先验
利用了何凯明的暗通道先验的实现过程。首先先将R,G,B格式转换从uchar到double，然后进行 归一化处理，使得取值范围在0-1之间，最后利用两个min得到Id。
```
Mat blue = Mat_<double>(Blue);
Mat green = Mat_<double>(Green);
Mat red = Mat_<double>(Red);
Mat Irn = red / 255;
Mat Ign = green / 255;
Mat Ibn = blue / 255;
Mat Id;
min(Irn, Ign, Id);
min(Id, Ibn, Id);
```
4. MSCN 系数的获得以及 rg 和 by 通道的获得
获得MSCN系数和cv。先制作MSCN_window的模板，然后对Ig和 MSCN_window进行实现线性空间滤波函数中的相关运算得到mu;同样利用Ig.*Ig和 MSCN_window的滤波运算再和mu_sq进行做差，最后开方得到sigma;利用(Ig- mu)./(sigma+1)得到MSCN系数。rg和by通道获得是利用OpenCV函数进行常规的矩 阵运算即可。
```
// MSCN
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
// rg 和 by 通道
Mat R, G, B;
Red.convertTo(R, CV_64F);
Green.convertTo(G, CV_64F);
Blue.convertTo(B, CV_64F);
Mat rg = R - G;
Mat by = 0.5*(R + G) - B;
```
### 雾感知统计特征的提取
f1,f2...f12共12个雾感统计特征，具体功能可以参考表格2.1
![image.png](https://upload-images.jianshu.io/upload_images/10826585-a2793242cb914c9c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* f1:

(1)概述:f1的提取，f1的功能可以参考表格2.1。
```
Mat tempbb = im2col(MSCN, ps);
Mat tempcc = var(tempbb);//
Mat MSCN_var = NewReshape(tempcc, row / ps);
```
(2)im2col的功能:重排矩阵列;
```
Mat im2col(Mat &InputMat, int ps)
{ int row = InputMat.rows / ps;
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
for (int i = 0; i < col; I++)
for (int j = 0; j < row; j++)
{ xin[j][i] = InputMat(Rect(i*ps, j*ps, ps, ps));
t[j][i] = xin[j][i].clone();
Mat temp = t[j][i].t();
t1[j][i] = temp.reshape(0, ps*ps);}
Mat outputMat = Mat::zeros(ps*ps, row*col, CV_64F);
Mat *t2 = new Mat[row*col];
int i, j, count = 0;
for (i = 0; i < col; I++)
{
for (j = 0; j < row; j++)
{
t2[count] = t1[j][I];
count++;
}
}
for (int i = 0; i < outputMat.rows; I++)
for (int j = 0; j < outputMat.cols; j++)
outputMat.at<double>(i, j) = t2[j].at<double>(i, 0);
return outputMat;}
```
(3)nanvar的功能:忽视掉NaN的方差的计算，注意分母是n-1;

(4)reshape的功能:重新调整矩阵的行数、列数。
```
Mat NewReshape(Mat &InputMat, int ps)
{ int row = 1;
int col = InputMat.cols / ps;
Mat *t = new Mat[col];
Mat *t1 = new Mat[col];
Mat *xin = new Mat[col];
for (int i = 0; i < col; I++)
{ xin[i] = InputMat(Rect(i*ps, 0, ps, 1));
t[i] = xin[i].clone();
t1[i] = t[i].reshape(0, ps);}
Mat outputMat = Mat::zeros(ps, row*col, CV_64F);
for (int i = 0; i < outputMat.rows; I++)
for (int j = 0; j < outputMat.cols; j++)
outputMat.at<double>(i, j) = t1[j].at<double>(i, 0);
return outputMat;}
```

* 雾感知统计特征中 f2 和 f3 的提取

(1)概述:f2和f3的提取，详情请参考以下代码。
```
Mat temp_vertical = circshift(MSCN, 1);
Mat temp_vertical2 = MSCN.mul(temp_vertical);
Mat MSCN_V_pair_col = im2col(temp_vertical2, ps);
Mat MSCN_V_pair_col_temp1 = MSCN_V_pair_col.clone();
for (int i = 0; i < MSCN_V_pair_col_temp1.rows; I++)
for (int j = 0; j < MSCN_V_pair_col_temp1.cols; j++)
if (MSCN_V_pair_col_temp1.at<double>(i, j)>0)
MSCN_V_pair_col_temp1.at<double>(i, j) = 0;
Mat MSCN_V_pair_col_temp2 = MSCN_V_pair_col.clone();
for (int i = 0; i < MSCN_V_pair_col_temp2.rows; I++)
for (int j = 0; j < MSCN_V_pair_col_temp2.cols; j++)
if (MSCN_V_pair_col_temp2.at<double>(i, j)<0)
MSCN_V_pair_col_temp2.at<double>(i, j) = 0;
Mat MSCN_V_pair_L_var=NewReshape(nanvar(MSCN_V_pair_col_temp1),row/ps);
Mat MSCN_V_pair_R_var=NewReshape(nanvar(MSCN_V_pair_col_temp2),row/ ps);
``` 
(2)cirshift: 矩阵循环平移。因为整个项目只用到了一次循环移位，所以该函数是特殊的 循环移位，也就是按列方向循环向下移动一位的函数。

* 雾感知统计特征的 f4 和 f5 的提取
(1)概述:f4和f5的提取，使用的算法和f2、f3相似。即reshape配合mean和 im2col的函数。
(2)mean:获取矩阵中每一列的平均值。
* 雾感知统计特征的 f6、f7 和 f8 的提取
(1)概述:本模块包括CE的编写，CE中又使用了border_in和border_out函数。 
(2)CE:感知对比能，获取灰度、蓝黄、红绿颜色通道。 
(3)border_in: 边界增加图像。 
(4)border_out:边界平衡图像。
* 雾感知统计特征的 f9 、f10 和 f11 的提取
(1)概述:f9、f10、f11的提取。
(2)Entropy:实现图像熵的功能。
Entropy的即图像熵-sum(p.*log2(p))的实现算法:
a. 利用temp[256]数组做一个类似统计表的功能，统计图像对应的0-255灰度值对 应的像素个数;
b. 计算每一个个像素的概率;
c. 根据定义计算图像熵。
```
double Entropy(Mat img)
{ double temp[256];
for (int i = 0; i<256; I++)
temp[i] = 0.0;
// 计算每个像素的累积值
for (int m = 0; m<img.rows; m++)
{ double *t = img.ptr<double>(m);
for (int n = 0; n<img.cols; n++)
{ int i = t[n];
temp[i] = temp[i] + 1;}
}
// 计算每个像素的概率
for (int i = 0; i<256; I++)
{temp[i] = temp[i] / (img.rows * img.cols);}
double result = 0;
// 计算图像熵
for (int i = 0; i<256; I++)
{ if (temp[i] == 0.0)
result = result;
else
result = result - temp[i] * (log(temp[i]) / log(2.0));
}
return result; }
```
(3)num2cell:将矩阵中的数据转成一个一个的孢元。 
(4)cellfun:对每个孢元单独计算.
* 雾感知统计特征的 f12 的提取
功能
概述:对以上f1—f12所有的特征进行重排矩阵列，然后放到一个矩阵feat中, 最后对feat进行变化得所要的值。

### MVG 模型的距离计算
概述:MVG模型Df的计算和Dff的计算，下面将主要介绍Df的计算，Dff的计算 除了数据不同外，基本上就是Df的重复。
* 自然无雾图像特征的提取和块的测试参数提取
自然无雾图像特征的提取:载入两个矩阵，cov_fogfreeparam和 mu_fogfreeparam块的测试参数提取:对之前的feat进行一些信息的提取。
* 距离计算
(1)概述:实现距离计算，包括了很多中间步骤
* 雾霾等级计算
(1)概述:Df和Df_map的获得，这个部分的完结。
```
Mat** distance_patch_t1 = new Mat *[mu_matrix.rows];
for (int i = 0; i<mu_matrix.rows; ++i)
distance_patch_t1[i] = new Mat[1];
for (int i = 0; i < mu_matrix.rows; I++)
for (int j = 0; j < 1; j++)
distance_patch_t1[i][j] = mu_cell[i] * (cov_cell[i][j].inv());
Mat** distance_patch_t2 = new Mat *[mu_matrix.rows];
for (int i = 0; i<mu_matrix.rows; ++i)
distance_patch_t2[i] = new Mat[1];
for (int i = 0; i < mu_matrix.rows; I++)
for (int j = 0; j < 1; j++)
distance_patch_t2[i][j]=distance_patch_t1[i][j]* (mu_transpose_cell[I]);
Mat distance_patch =
Mat::zeros(distance_patch_t2[0][0].rows*mu_matrix.rows,
distance_patch_t2[0][0].cols, CV_64F);
for (int i = 0; i < distance_patch.rows; I++)
for (int j = 0; j < distance_patch.cols; j++)
{
distance_patch.at<double>(i,j)=distance_patch_t2[i][0].at<double>(0,j);
}
sqrt(distance_patch, distance_patch);
Mat Df = NewMean(distance_patch);
Mat distance_patch_t = distance_patch.t();
Mat Df_map = NewReshape(distance_patch_t, row / ps);
```
###雾霾程度的获取
概述:首先获取雾霾密度值，然后根据雾霾密度值计算雾霾程度。即先利用D = Df / (Dff + 1)获取雾霾密度值，然后根据雾霾密度值D在0 ≤ D < 1为低，1 < D ≤ 3为中，雾霾密度值D在D > 3为高，进行雾霾程度输出。

#五. 测试结果
* 首先我们载入一张雾霾程度比较低的图片
![0FC0084E60774D5FF1CC0F5D46650C8B.jpg](https://upload-images.jianshu.io/upload_images/10826585-14f46bd44c1bda39.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
* 再载入一张雾霾程度较高的图片
![D875521A9C5C3638B3CE2B1B35A02C5B.jpg](https://upload-images.jianshu.io/upload_images/10826585-bdb7cd37b2187073.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
通过以上分析，我们可以判断这个算法输出的客观评价和人的主观评价相符，大家也可下载demo载入一些其他的图片测试，雾霾程度客观评价值越高，说明雾霾程度越严重。


#六. 遇到的坑以及处理方法
1. Xcode和Visual Studio毕竟是不同平台，iOS对于OpenCV的支持不如VS那么简单，不能做到一行代码调用多个函数
```
//像下面这种代码VS中可以运行，但是Xcode就不行，需要将一个函数拆分为多个函数
Mat MSCN_V_pair_L_var=NewReshape(nanvar(MSCN_V_pair_col_temp1),row/ps);
```
```
//举个例子，Xcode中需要一步一步
temp=nanvar(MSCN_V_pair_col_temp1)；
Mat MSCN_V_pair_L_var=NewReshape（temp，row/ps）；
```
2. VS中很多函数调用using namespace cv就可以了，但是Xcode的对这个的支持似乎不好，很多函数都要加上cv::
```
//Xcode需要以下的写法，VS的rect不需要加cv了
Mat outputMat = InputMat(cv::Rect(0, 0, patch_col_num * ps, patch_row_num * ps));
```
3. iOS 真机测试中，iOS的机器在载入图片的时候容易报内存不足导致闪退，这个在模拟器上不会出现这个问题。这个问题原因存在于，iOS照相保存的图片文件过大，导致这个算法需要运行耗的内存过多，所以我对于这个解决办法是先压缩图片，然后在将图片进行之后的处理。之后在iPhone 5s和7上测试后都可以完美运行了。
```
//图像压缩
cv::resize(inputMat, tmp, cv::Size(inputMat.rows / 2, inputMat.cols/ 2));
```
4. 运行算法的时候需要一些时间，导致app有点卡
解决方法是利用了GCD处理多线程，将处理算法的程序运行在线程中，在运行完成后再回到主线程中更新UI.
```
 dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    // 获取主队列
    dispatch_queue_t mainQueue = dispatch_get_main_queue();
    dispatch_async(queue, ^{
        // 异步追加任务
        //耗时操作放在这里
        //算法利用
        uint64_t time = dispatch_benchmark(1, ^{
            self.fogValue=[Fogdetect Fogdetecting:self.imageView.image];
        });
        NSLog(@"%lf",self.fogValue);
        NSLog(@"耗时 ---> %llu ns",time);
        
        // 回到主线程
        dispatch_async(mainQueue, ^{
            // 追加在主线程中执行的任务
            self.value.text=[NSString stringWithFormat:@"%lf",self.fogValue]  ;
            if (self.fogValue > 3)
            self.Degree.text = @"高";
            else if (self.fogValue > 1 && self.fogValue <= 3)
            self.Degree.text = @"中";
            else
            self.Degree.text = @"低";
            NSLog(@"回到主线程");
        });
    });
```
# 总结
客观评价的目标使得评价模型准确地反映人眼视觉感知的主观评价。本课题研究基于自然场景统计特征(NSS)和其他感知特征的雾霾浓度评价研究，使得对于输入的图像，准确输出评价值。测试表明，客观评价的值符合人的主观评价，客观评价值越高，雾霾程度越高。

源代码不包含 opencv2.framework，请自行下载后添加进项目中。
最后，附上demo地址https://github.com/echooj/FogDetect，希望大家帮忙点个赞
该demo目前没有做屏幕适配，原因是本人有点懒-_-，请大家用4.7寸的设备运行，有任何问题可以简书私戳我，谢谢大家。
>本文算法的主要参考文章
[1] Lark Kwon Choi, Jaehee You, and Alan Conrad Bovik, Referenceless Prediction of Perceptual Fog Density. TIP, 2015.
[2] and Perceptual Image DefoggingAnish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-Reference Image Quality Assessment in the Spatial Domain. TIP, 2012.
[3] Rafael C.Gonzalez，Richard E.Woods，Steven L.Eddins 著，阮秋琦等译.数字图像处 理(MATLAB 版)[M].北京:电子工业出版社，2005:58-60.
[4]基于 OpenCV 的 iOS 客户端答题卡识别算法https://www.jianshu.com/p/eed90371a3a6
>