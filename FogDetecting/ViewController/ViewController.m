//
//  ViewController.m
//  FogDetecting
//
//  Created by J Echoo on 2018/5/21.
//  Copyright © 2018年 J.Echoo. All rights reserved.
//
#import "ViewController.h"
#import "Fogdetect.h"
#import "HXPhotoPicker.h"
extern uint64_t dispatch_benchmark(size_t count, void (^block)(void));
@interface ViewController ()<HXAlbumListViewControllerDelegate>
@property (weak, nonatomic) IBOutlet UILabel *value;
@property (weak, nonatomic) IBOutlet UILabel *Degree;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (nonatomic,assign) CGFloat fogValue;
@property (strong, nonatomic) HXPhotoManager *manager;
@property (strong, nonatomic) HXDatePhotoToolManager *toolManager;
@end

@implementation ViewController
- (IBAction)Evaluate:(id)sender {
    
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
 
   
}
- (IBAction)convert:(id)sender {

//    uint64_t time = dispatch_benchmark(1, ^{
//      //  self.resultImageView.image = [OpenCVManager correctWithUIImage:[UIImage imageNamed:@"123456"]];
//       self.fogValue=[Fogdetect Fogdetecting:[UIImage imageNamed:@"test_image1"]];
//    });
//    NSLog(@"%lf",self.fogValue);
//    NSLog(@"耗时 ---> %llu ns",time);
    
    self.manager.configuration.saveSystemAblum = YES;
    
    __weak typeof(self) weakSelf = self;
    [self hx_presentAlbumListViewControllerWithManager:self.manager done:^(NSArray<HXPhotoModel *> *allList, NSArray<HXPhotoModel *> *photoList, NSArray<HXPhotoModel *> *videoList, BOOL original, HXAlbumListViewController *viewController) {
        if (photoList.count > 0) {
            //            HXPhotoModel *model = photoList.firstObject;
            //            weakSelf.imageView.image = model.previewPhoto;
            [weakSelf.view showLoadingHUDText:@"获取图片中"];
            [weakSelf.toolManager getSelectedImageList:photoList requestType:0 success:^(NSArray<UIImage *> *imageList) {
                [weakSelf.view handleLoading];
                weakSelf.imageView.image = imageList.firstObject;
                
            } failed:^{
                [weakSelf.view handleLoading];
                [weakSelf.view showImageHUDText:@"获取失败"];
            }];
            NSSLog(@"%ld张图片",photoList.count);
        }else if (videoList.count > 0) {
            [weakSelf.toolManager getSelectedImageList:allList success:^(NSArray<UIImage *> *imageList) {
                weakSelf.imageView.image = imageList.firstObject;
            } failed:^{
                
            }];
            // 通个这个方法将视频压缩写入临时目录获取视频URL  或者 通过这个获取视频在手机里的原路径 model.fileURL  可自己压缩
            [weakSelf.view showLoadingHUDText:@"视频写入中"];
            [weakSelf.toolManager writeSelectModelListToTempPathWithList:videoList success:^(NSArray<NSURL *> *allURL, NSArray<NSURL *> *photoURL, NSArray<NSURL *> *videoURL) {
                NSSLog(@"%@",videoURL);
                [weakSelf.view handleLoading];
            } failed:^{
                [weakSelf.view handleLoading];
                [weakSelf.view showImageHUDText:@"写入失败"];
                NSSLog(@"写入失败");
            }];
            NSSLog(@"%ld个视频",videoList.count);
        }
    } cancel:^(HXAlbumListViewController *viewController) {
        NSSLog(@"取消了");
    }];
}

- (HXPhotoManager *)manager {
    if (!_manager) {
        _manager = [[HXPhotoManager alloc] initWithType:HXPhotoManagerSelectedTypePhotoAndVideo];
        _manager.configuration.singleSelected = YES;
        _manager.configuration.albumListTableView = ^(UITableView *tableView) {
            //            NSSLog(@"%@",tableView);
        };
        _manager.configuration.singleJumpEdit = YES;
        _manager.configuration.movableCropBox = YES;
        _manager.configuration.movableCropBoxEditSize = YES;
        //        _manager.configuration.movableCropBoxCustomRatio = CGPointMake(1, 1);
    }
    return _manager;
}

- (HXDatePhotoToolManager *)toolManager {
    if (!_toolManager) {
        _toolManager = [[HXDatePhotoToolManager alloc] init];
    }
    return _toolManager;
}

- (void)albumListViewController:(HXAlbumListViewController *)albumListViewController didDoneAllList:(NSArray<HXPhotoModel *> *)allList photos:(NSArray<HXPhotoModel *> *)photoList videos:(NSArray<HXPhotoModel *> *)videoList original:(BOOL)original {
    if (photoList.count > 0) {
        HXPhotoModel *model = photoList.firstObject;
        self.imageView.image = model.previewPhoto;//
  
        NSSLog(@"%ld张图片",photoList.count);
    }else if (videoList.count > 0) {
        __weak typeof(self) weakSelf = self;
        [self.toolManager getSelectedImageList:allList success:^(NSArray<UIImage *> *imageList) {
            weakSelf.imageView.image = imageList.firstObject;
            
            
        } failed:^{
            
        }];
        
        // 通个这个方法将视频压缩写入临时目录获取视频URL  或者 通过这个获取视频在手机里的原路径 model.fileURL  可自己压缩
        [self.view showLoadingHUDText:@"视频写入中"];
        [self.toolManager writeSelectModelListToTempPathWithList:videoList success:^(NSArray<NSURL *> *allURL, NSArray<NSURL *> *photoURL, NSArray<NSURL *> *videoURL) {
            NSSLog(@"%@",videoURL);
            [weakSelf.view handleLoading];
        } failed:^{
            [weakSelf.view handleLoading];
            [weakSelf.view showImageHUDText:@"写入失败"];
            NSSLog(@"写入失败");
        }];
        NSSLog(@"%ld个视频",videoList.count);
    }
}


- (void)viewDidLoad {
    [super viewDidLoad];
  //  self.imageView.image = [UIImage imageNamed:@"test_image1"];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
