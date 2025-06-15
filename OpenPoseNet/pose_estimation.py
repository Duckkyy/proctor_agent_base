from PIL import Image
import cv2
import numpy as np
from matplotlib import cm, pyplot as plt
import time
import torch
import sys, os, math

# Use relative imports for local modules
from util.decode_pose import decode_pose
from util.openpose_net import OpenPoseNet

# net = OpenPoseNet()

class PoseEstimation:
    def __init__(self):
        self.net = OpenPoseNet()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        weight_file_path = os.path.join(self.current_dir, 'weights', 'pose_model_scratch.pth')
        # Update torch.load with weights_only=True
        self.net_weights = torch.load(
            weight_file_path, 
            map_location={'cuda:0': 'cpu'},
            weights_only=True  # Add this parameter
        )
        keys = list(self.net_weights.keys())
        # print(keys)
        weights_load = {}

        for i in range(len(keys)):
            weights_load[list(self.net.state_dict().keys())[i]
                        ] = self.net_weights[list(keys)[i]]

        state = self.net.state_dict()
        state.update(weights_load)
        self.net.load_state_dict(state)

        print('Pose Estimation Model Loaded')

    def estimate_pose(self, oriImg):

        # oriImg = cv2.imread(image_path)
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    
        # 画像のリサイズ
        size = (368, 368)
        img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)

        # 画像の前処理
        img = img.astype(np.float32) / 255.

        # 色情報の標準化
        color_mean = [0.485, 0.456, 0.406]
        color_std = [0.229, 0.224, 0.225]

        # 色チャネルの順番を誤っています
        preprocessed_img = img.copy()  # RGB

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        # （高さ、幅、色）→（色、高さ、幅）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

        # 画像をTensorに
        img = torch.from_numpy(img)

        # ミニバッチ化：torch.Size([1, 3, 368, 368])
        x = img.unsqueeze(0)

        # OpenPoseでheatmapsとPAFsを求めます
        self.net.eval()
        predicted_outputs, _ = self.net(x)

        # 画像をテンソルからNumPyに変化し、サイズを戻します
        pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
        heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

        pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

        pafs = cv2.resize(
            pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(
            heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        start = time.perf_counter()
        _, result_img, joint_list, person_to_joint_assoc = decode_pose(oriImg, heatmaps, pafs)
        # print("ESTIMATE TIME: ", time.perf_counter() - start, result_img.shape)

        # 結果を描画
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        # cv2.resize(result_img, (result_img.shape[0]*3, oriImg.shape[1]*3), interpolation = cv2.INTER_AREA)

        # cv2.imwrite(output_image_path, result_img)
        return result_img, joint_list.tolist(), person_to_joint_assoc.tolist(), heatmaps, pafs
    
    def image_pose_estimation(self, image_path):
        output_path = os.path.join(self.current_dir, "result.jpg")
        orig_image = cv2.imread(image_path)
        result_img, joint_list, person_to_joint_assoc, heatmaps, pafs = self.estimate_pose(orig_image)

        # Save the result image
        cv2.imwrite(output_path, result_img)

        return joint_list, person_to_joint_assoc

    def video_pose_estimation(self, video_path, save_npz=False):  # NOT DONE YET
        output_path = os.path.join(self.current_dir, "result.mp4")
        cap = cv2.VideoCapture(video_path)
        # Get video properties for creating the output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create a VideoWriter object to write the video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        all_coco17_keypoints = []

        # BODY_25 to COCO_17 joint index mapping
        body25_to_coco17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        frame_cnt = 0

        while cap.isOpened():
            frame_cnt += 1
            print(frame_cnt)
            ret, frame = cap.read()
            if not ret:
                break

            result_frame, joint_list, person_to_joint_assoc, _, _ = self.estimate_pose(frame)

            # Write the frame into the output video
            out.write(result_frame)

            if joint_list and len(joint_list) >= 17:
                # Extract only (x, y) from joint_list
                keypoints = np.array([[joint[0], joint[1]] for joint in joint_list])  # [25, 2]
                keypoints_coco17 = keypoints[body25_to_coco17]  # [17, 2]
            else:
                keypoints_coco17 = np.zeros((17, 2))  # fallback for missing detection

            all_coco17_keypoints.append(keypoints_coco17)

        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save as .npz
        if save_npz:
            keypoints_array = np.stack(all_coco17_keypoints)  # [frames, 17, 2]
            positions_2d = {
                'S1': {
                    'custom': keypoints_array.astype(np.float32)
                }
            }

            npz_file_path = os.path.join(self.current_dir, "result.npz")
            np.savez_compressed(npz_file_path, positions_2d=positions_2d)

        return joint_list, person_to_joint_assoc

    def estimate_and_evaluate_pose(self, video_path): 
        cap = cv2.VideoCapture(video_path)
        # Get video properties for creating the output video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total_frames , fps)

        cut_frame = [[(290,650),(315,650)], [(315,650),(980,1305)], [(215,525),(690,950)], [(230,525),(1275,1550)]]
        angle_ls = [[],[],[],[]]
        time_render = []
        frame_cnt = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for i in range(len(angle_ls)):
                person_frame = frame[cut_frame[i][0][0]:cut_frame[i][0][1],cut_frame[i][1][0]:cut_frame[i][1][1]]
                result_img, joint_list, person_to_joint_assoc = self.estimate_pose(person_frame)
                nose_idx = int(person_to_joint_assoc[0][0])
                neck_idx = int(person_to_joint_assoc[0][1])
                # x = [math.ceil(float(time)) for time in person_to_joint_assoc]
                # x = [math.ceil(float(time)) for time in x]
                nose_kpt = joint_list[nose_idx][:2]
                neck_kpt = joint_list[neck_idx][:2]

                angle_ls[i].append(np.linalg.norm(np.array([neck_kpt[0] - nose_kpt[0], neck_kpt[1] - nose_kpt[1]])))
                # angle_ls[i].append(self.calculate_angle(nose_kpt, neck_kpt))
                # print(np.linalg.norm(vector))
            time_render.append(fps * frame_cnt)
            frame_cnt += 1
            if frame_cnt % 10 == 0:
                print("ON FRAME ", frame_cnt)

        # Release everything when job is finished
        cap.release()
        # out.release()
        cv2.destroyAllWindows()

        return angle_ls, time_render
                

pose_estimator = PoseEstimation()
video_path = '/home/dh11255z/Documents/proctor_agent_base/test.mp4'
_, _ = pose_estimator.video_pose_estimation(video_path)
# res_frame = pose_estimator.estimate_pose_and_checking(cv2.imread('/home/dai/MCGaze/IMG_4721.jpg'))
# cv2.imwrite('/home/dai/MCGaze/IMG_4721_res.jpg', res_frame)
# cap = cv2.VideoCapture('/home/dai/MCGaze/IMG_4724.mp4')
# Get video properties for the output video
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(width, height, fps, frame_count)

# Prepare the output video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
# out = cv2.VideoWriter('/home/dai/MCGaze/IMG_4724_res2.mp4', fourcc, fps, (width, height))
# frame_cnt = 0
# while True:
    # Read frame by frame
    # ret, frame = cap.read()

    # Check if frame is read correctly
    # if not ret:
    #     break
    
    # res_frame = pose_estimator.estimate_pose_and_checking(frame)

    # Write the frame to the output video
    # out.write(res_frame)
    # if frame_cnt % 10 == 0:
    #     print("END FRAME: ", frame_cnt)
    # frame_cnt += 1

# Release resources
# cap.release()
# out.release()
# _, _ = pose_estimator.video_pose_estimation('/home/dai/MCGaze/IMG_4724.mp4')




# image_path = '/home/dh11255z/Documents/computer-rendering/test.png'
# orig_image = cv2.imread(image_path)
# res, _, _, _, _ = pose_estimator.estimate_pose(orig_image)
# res, joint_list, person_to_joint_assoc, heatmaps, pafs = pose_estimator.estimate_pose(orig_image)
# cv2.imwrite('/home/dh11255z/Documents/computer-rendering/res.png', res)
# print("JOINT LIST: ", joint_list)
# print("PERSON TO JOINT ASSOC: ", person_to_joint_assoc)
# print("HEATMAPS: ", heatmaps)
# print("PAFS: ", pafs)
# print(NG_boxes)
# joint_list, person_to_joint_assoc = pose_estimator.video_pose_estimation("/home/dai/MCGaze/IMG_4721.mp4")






# distane_ls, time_render = pose_estimator.estimate_and_evaluate_pose("/home/dai/MCGaze/IMG_4721.mp4")

# colors = ['red', 'green', 'blue', 'orange']
# subgroups = ["Person 1", "Person 2", "Person 3", "Person 4"]

# fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# axs_flat = axs.flatten()

# for i, ax in enumerate(axs_flat):
#     categories = [math.ceil(float(time)) for time in time_render]
#     values = distane_ls[i]
#     title = subgroups[i]
#     color = colors[i]
    
#     ax.scatter(categories, values, color=color)
#     # ax.axhline(0, color='lightblue', linewidth=2, linestyle='-')
    
#     ax.set_title(title)
#     fig.suptitle('Distance betwwen nose and neck keyponts', fontsize=20, fontweight='bold')
#     # fig.suptitle('Angle of nose & neck keypoint and y-axis', fontsize=20, fontweight='bold')

#     ax.set_xlabel('Time(s)')
#     ax.set_ylabel('Distance')
#     # ax.set_ylabel('Angle')

#     # Add a small box of information (text annotation)
#     data = np.array(distane_ls[i])
#     mean = np.mean(data)
#     std_dev = np.std(data)
#     plt.annotate(f'Mean: {mean}\nStd: {std_dev}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

# plt.tight_layout()

# plt.savefig('analysis/nose_neck_distance.png')
# # plt.savefig('analysis/nose_neck_angle.png')


# pose_info = {"kpts_info": joint_list, "people_kpts_info": person_to_joint_assoc}
# gaze_info = np.array([1,2,3])
# checkor = SuspiciousChecking(pose_info, gaze_info, image_path)
# checkor.passing_material()
# checkor.using_phone()
# cv2.imwrite('/home/dai/MCGaze/test_1.jpg',result_img)

# def create_heatmap_img(oriImg, heatmaps, pafs):
#     # 左肘と左手首のheatmap、そして左肘と左手首をつなぐPAFのxベクトルを可視化する
#     # 左肘
#     heat_map = heatmaps[:, :, 6]  # 6は左肘
#     heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
#     heat_map = np.asarray(heat_map.convert('RGB'))

#     # 合成して表示
#     blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)

#     # 左手首
#     heat_map = heatmaps[:, :, 7]  # 7は左手首
#     heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
#     heat_map = np.asarray(heat_map.convert('RGB'))

#     # 合成して表示
#     blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)

#     # 左肘と左手首をつなぐPAFのxベクトル
#     paf = pafs[:, :, 24]
#     paf = Image.fromarray(np.uint8(cm.jet(paf)*255))
#     paf = np.asarray(paf.convert('RGB'))

#     # 合成して表示
#     blend_img = cv2.addWeighted(oriImg, 0.5, paf, 0.5, 0)