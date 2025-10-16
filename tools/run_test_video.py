import cv2
import numpy as np
import torch
from os.path import isfile, join
from tools.test import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--video', default='test.mp4', help='path to input video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # 读取视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    # 读取第一帧选择ROI
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频第一帧")
        exit()

    cv2.namedWindow("SiamMask", cv2.WINDOW_NORMAL)
    init_rect = cv2.selectROI('SiamMask', first_frame, False, False)
    x, y, w, h = init_rect

    # 初始化跟踪器
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = siamese_init(first_frame, target_pos, target_sz, siammask, cfg['hp'], device=device)

    toc = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tic = cv2.getTickCount()
        if frame_id > 0:
            state = siamese_track(state, frame, mask_enable=True, refine_enable=True, device=device)
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            cv2.polylines(frame, [location.astype(int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', frame)
            key = cv2.waitKey(1)
            # if key > 0:  # 按任意键退出
            #     break

        toc += cv2.getTickCount() - tic
        frame_id += 1

    toc /= cv2.getTickFrequency()
    fps = frame_id / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visualization!)'.format(toc, fps))
    cap.release()
    cv2.destroyAllWindows()
