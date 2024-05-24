import matplotlib.pyplot as plt

if __name__ == '__main__':
    # plt.rcParams['axes.labelweight'] = 'bold'
    ray_iou_1m = [31.2, 31.3, 32.0]
    ray_iou_2m = [37.2, 37.5, 38.3]
    ray_iou_4m = [41.1, 41.5, 42.4]
    x = [1, 2, 3]
    # plot scatter plot
    plt.figure()
    plt.scatter(x, ray_iou_1m, label='RayIoU_1m')
    plt.scatter(x, ray_iou_2m, label='RayIoU_2m', marker='x')
    plt.scatter(x, ray_iou_4m, label='RayIoU_4m', marker='^')
    plt.plot(x, ray_iou_1m)
    plt.plot(x, ray_iou_2m)
    plt.plot(x, ray_iou_4m)
    # show data, text front and bold
    for i in range(3):
        plt.text(x[i], ray_iou_1m[i], ray_iou_1m[i], ha='center', va='bottom', fontsize=12)
        plt.text(x[i], ray_iou_2m[i], ray_iou_2m[i], ha='center', va='bottom', fontsize=12)
        plt.text(x[i], ray_iou_4m[i], ray_iou_4m[i], ha='center', va='bottom', fontsize=12)
    # different scatter different type

    # move legend to correct position
    # move legend lower
    # axis-x interval set to 1
    plt.xticks(range(1, 4), fontsize='large')
    plt.yticks(fontsize='large')

    plt.gca().set_facecolor('white')
    plt.grid(color='black', linewidth=0.1)

    legend = plt.legend(loc='center left', bbox_to_anchor=(0, 0.3), fontsize=14)
    plt.setp(legend.get_texts())
    plt.xlabel('Number of Stages', fontsize='large')
    plt.savefig('ray_casting_iou.jpg', dpi=300)
    # front bigger
    plt.show()