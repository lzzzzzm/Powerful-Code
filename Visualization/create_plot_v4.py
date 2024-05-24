import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.rcParams['axes.labelweight'] = 'bold'
    ray_iou_1m = [32.2, 32.4, 32.6, 32.7, 32.6]
    ray_iou_2m = [38.5, 38.6, 38.8, 39.0, 38.9]
    ray_iou_4m = [42.3, 42.4, 42.6, 42.7, 42.5]
    gpu = [4.82, 4.85, 4.89, 5.01, 5.06, 5.07, 5.2]
    ray_number = [48, 64, 80, 96, 112]
    plt.figure()
    plt.scatter(ray_number, ray_iou_1m, label='RayIoU_1m')
    plt.scatter(ray_number, ray_iou_2m, label='RayIoU_2m', marker='x')
    plt.scatter(ray_number, ray_iou_4m, label='RayIoU_4m', marker='^')
    plt.plot(ray_number, ray_iou_1m)
    plt.plot(ray_number, ray_iou_2m)
    plt.plot(ray_number, ray_iou_4m)
    legend = plt.legend(loc='center left', bbox_to_anchor=(0, 0.3))
    plt.setp(legend.get_texts(), fontweight='bold')
    plt.xlabel('Number of Sample Point', fontsize='large')
    plt.xticks(fontsize='large', fontweight='bold')
    plt.yticks(fontsize='large', fontweight='bold')
    plt.grid()
    # for i in range(3):
    #     plt.text(gpu[i], ray_iou_1m[i], ray_iou_1m[i], ha='center', va='bottom', fontsize=12, fontweight='bold')
    #     plt.text(gpu[i], ray_iou_2m[i], ray_iou_2m[i], ha='center', va='bottom', fontsize=12, fontweight='bold')
    #     plt.text(gpu[i], ray_iou_4m[i], ray_iou_4m[i], ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.show()