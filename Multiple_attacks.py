import os
import shutil
import json
import matplotlib.pyplot as plt
from attack import Attack, GradCAMWrapper, load_images, load_model
from utils import images_to_video

image_folder = 'imagenet'
model_name = 'ViT' 
max_loop = 1000
load_images_num = 36 # 读取的图片数量

# config_dict = dict(
#   mask=['cam', 'grad', 'random'], 
#   attacked_pixel = [20000], 
#   noise_ratios = [1],
#   ratio_of_succeeds = [0.5]
# )

config_dict = dict(
  mask=['cam', 'grad', 'random'], # 攻击方式
  attacked_pixel = [10, 100, 500, 1000, 2000, 5000, 10000], # 攻击像素数
  noise_ratios = [0.5, 1], # 噪声强度,也就是生成白噪声的方差
  ratio_of_succeeds = [0.5, 0.8, 1] # class改变的比例
)
def save_result_json(root_path):
    results_dict = {}
    image_tmp_path = os.path.join(root_path, 'images_tmp')
    for mask in config_dict['mask']:
        results_dict[mask] = {}
        key = 0
        for attacked_pixel in config_dict['attacked_pixel']:
            for noise_ratio in config_dict['noise_ratios']:
                for ratio_of_succeed in config_dict['ratio_of_succeeds']:
                    result_dict = {}
                    result_dict['attacked_pixel'] = attacked_pixel
                    result_dict['ratio'] = noise_ratio
                    result_dict['ratio_of_succeeds'] = ratio_of_succeed
                    model, _, _ = load_model(model_name)
                    input_tensor, _, _ = load_images(model, image_folder, load_images_num = load_images_num)
                    gradcam = GradCAMWrapper(model_name)
                    output_path = os.path.join(image_tmp_path, f'{image_folder}_{model_name}_{mask}/{attacked_pixel}_noise{noise_ratio}_change{ratio_of_succeed}/')
                    loop_count, num_differences_list, original_classes, predicted_classes = Attack(gradcam, input_tensor, num=attacked_pixel, ratio=noise_ratio).predict_and_attack(ratio_of_succeed=ratio_of_succeed, mask=mask, max_loop=max_loop, output_path=output_path)
                    video_path = os.path.join(root_path, f'videos/{image_folder}_{model_name}_{mask}/{attacked_pixel}_noise{noise_ratio}_change{ratio_of_succeed}.avi')
                    result_dict['loop_count'] = loop_count
                    # result_dict['original_classes'] = original_classes
                    # result_dict['predicted_classes'] = predicted_classes
                    result_dict['output_path'] = output_path
                    results_dict[mask][key] = result_dict
                    key += 1
                    print(f"num_differences_list = {num_differences_list}")
                    images_to_video(output_path, video_path, num_differences_list)
                    print(f'删除{output_path}... ...')
                    shutil.rmtree(output_path)
    print(f"删除{image_tmp_path}... ...")
    shutil.rmtree(image_tmp_path)

    result_path = os.path.join(root_path, "result")
    os.makedirs(result_path, exist_ok=True)
    json_file_path = os.path.join(result_path, "result.json")

    with open(json_file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=2)
    print(f'Results saved to {os.path.join(root_path, "result.json")}')

def read_result_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        results_dict = json.load(json_file)
    return results_dict

def plot_curve(ax, results, mask, noise_ratio, ratio_of_succeed):
    data = results[mask]
    
    attacked_pixels = []
    loop_counts = []

    for key in data:
        result_dict = data[key]
        attacked_pixel = result_dict['attacked_pixel']
        loop_count = result_dict['loop_count']

        attacked_pixels.append(attacked_pixel)
        loop_counts.append(loop_count)

    ax.plot(attacked_pixels, loop_counts, label=f'{mask} Mask (Noise Ratio={noise_ratio}, Ratio of Succeeds={ratio_of_succeed})')


def plot_results(json_file_path):
    results = read_result_json(json_file_path)
    noise_ratios_to_plot = config_dict['noise_ratios']
    ratios_of_succeeds_to_plot = config_dict['ratio_of_succeeds']

    # 创建一个画布和子图
    fig, axs = plt.subplots(len(noise_ratios_to_plot), len(ratios_of_succeeds_to_plot), figsize=(15, 10))
    fig.suptitle('Loop Count vs Attacked Pixel for Different Noise Ratios and Ratio of Succeeds')

    for i, noise_ratio in enumerate(noise_ratios_to_plot):
        for j, ratio_of_succeed in enumerate(ratios_of_succeeds_to_plot):
            axs[i, j].set_title(f'Noise Ratio={noise_ratio}, Ratio of Succeeds={ratio_of_succeed}')
            axs[i, j].set_xlabel('Attacked Pixel')
            axs[i, j].set_ylabel('Loop Count')
            axs[i, j].grid(True)
    
            for mask in config_dict['mask']:
                plot_curve(axs[i, j], results, mask, noise_ratio, ratio_of_succeed)

    save_path = os.path.join(os.path.dirname(json_file_path), 'result_plot.png')
    print(f'Saving plot to {save_path}... ...')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.legend()
    plt.savefig(save_path)
    plt.close
   
if __name__ == '__main__':
    root_path = 'data'
    save_result_json(root_path)
    # json_file_path = os.path.join(root_path, 'result', 'result.json')
    # save_path = os.path.join(root_path, 'result', 'result_plot.png')
    # plot_results(json_file_path, save_path)
