//
//  HXCustomPreviewView.h
//  微博照片选择
//
//  Created by 洪欣 on 2017/10/31.
//  Copyright © 2017年 洪欣. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

@protocol HXCustomPreviewViewDelegate <NSObject>
@optional;
- (void)tappedToFocusAtPoint:(CGPoint)point;
- (void)pinchGestureScale:(CGFloat)scale;
- (void)didLeftSwipeClick;
- (void)didRightSwipeClick;
@end

@interface HXCustomPreviewView : UIView
@property (strong, nonatomic) AVCaptureSession *session;
@property (weak, nonatomic) id<HXCustomPreviewViewDelegate> delegate;

@property(nonatomic,assign) CGFloat beginGestureScale;
@property(nonatomic,assign) CGFloat effectiveScale;
@property(nonatomic,assign) CGFloat maxScale;

- (void)addSwipeGesture;

@property (nonatomic) BOOL tapToFocusEnabled;
@property (nonatomic) BOOL tapToExposeEnabled;
@property (nonatomic) BOOL pinchToZoomEnabled;
@end
