import matplotlib.pyplot as plt

if __name__ == '__main__':
    # plt.rcParams['axes.labelweight'] = 'bold'
    ray_iou_1m = [31.3, 31.6, 31.9, 32.1, 32.3, 32.5, 32.8]
    ray_iou_2m = [37.5, 37.7, 37.9, 38.3, 38.6, 38.9, 39.0]
    ray_iou_4m = [41.5, 41.8, 42.1, 42.2, 42.5, 42.8, 42.9]
    ray_iou_stage = [36.8,37.0,37.4,37.6,37.8,38.2,38.2]
                    # (1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2), (3, 2, 2), (3, 3, 2), (3, 3, 3)
    gpu = [4.82, 4.85, 4.89, 5.01, 5.06, 5.07, 5.2]
    plt.figure()
    plt.scatter(gpu, ray_iou_1m, label='RayIoU_1m')
    plt.scatter(gpu, ray_iou_2m, label='RayIoU_2m', marker='x')
    plt.scatter(gpu, ray_iou_4m, label='RayIoU_4m', marker='^')
    plt.plot(gpu, ray_iou_1m)
    plt.plot(gpu, ray_iou_2m)
    plt.plot(gpu, ray_iou_4m)
    legend = plt.legend(loc='center left', bbox_to_anchor=(0, 0.3), fontsize=14)

    plt.setp(legend.get_texts())
    plt.xlabel('Memory(G)', fontsize=14)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.grid()
    # for i in range(3):
    #     plt.text(gpu[i], ray_iou_1m[i], ray_iou_1m[i], ha='center', va='bottom', fontsize=12, fontweight='bold')
    #     plt.text(gpu[i], ray_iou_2m[i], ray_iou_2m[i], ha='center', va='bottom', fontsize=12, fontweight='bold')
    #     plt.text(gpu[i], ray_iou_4m[i], ray_iou_4m[i], ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig('memory.jpg', dpi=300)
    plt.show()