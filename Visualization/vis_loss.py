import os
import mmcv
import mmengine
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # plt.style.use('seaborn-darkgrid')
    # plt.rcParams['axes.labelweight'] = 'bold'
    # Load the log file
    log_file = 'demo_loss_data/fb-occ.log'
    our_log_file = 'demo_loss_data/ours.log'
    with open(log_file, 'r') as f:
        lines = f.readlines()
    with open(our_log_file, 'r') as f:
        our_lines = f.readlines()
    fb_loss_voxel_sem_sacle_c_0 = []
    ours_loss_voxel_sem_sacle_c_0 = []
    for line in lines:
        if 'loss_voxel_sem_scal_c_0' in line:
            # find the line with loss_voxel_sem_scal_c_0 and the loss value
            line = line.strip()
            loss_voxel_sem_sacle_c_0_index = line.find('loss_voxel_sem_scal_c_0')
            loss_voxel_sem_sacle_c_0_value = float(line[loss_voxel_sem_sacle_c_0_index + 24: loss_voxel_sem_sacle_c_0_index + 30])
            fb_loss_voxel_sem_sacle_c_0.append(loss_voxel_sem_sacle_c_0_value)
    for line in our_lines:
        if 'loss_voxel_sem_scal_c_0' in line:
            # find the line with loss_voxel_sem_scal_c_0 and the loss value
            line = line.strip()
            loss_voxel_sem_sacle_c_0_index = line.find('loss_voxel_sem_scal_c_0')
            loss_voxel_sem_sacle_c_0_value = float(line[loss_voxel_sem_sacle_c_0_index + 24: loss_voxel_sem_sacle_c_0_index + 30])
            ours_loss_voxel_sem_sacle_c_0.append(loss_voxel_sem_sacle_c_0_value)
    # Check the length of the loss list
    print(len(fb_loss_voxel_sem_sacle_c_0))
    print(len(ours_loss_voxel_sem_sacle_c_0))
    # make same length
    ours_loss_voxel_sem_sacle_c_0 = ours_loss_voxel_sem_sacle_c_0[:len(fb_loss_voxel_sem_sacle_c_0)]

    # Plot the loss curve
    plt.figure()
    plt.plot(fb_loss_voxel_sem_sacle_c_0, label='Dense')
    plt.plot(ours_loss_voxel_sem_sacle_c_0, label='Sparse')

    # set background to white colo
    plt.gca().set_facecolor('white')
    # show gird
    plt.grid(color='black', linewidth=0.1)


    # make legend front bigger
    legend = plt.legend(fontsize=16)
    plt.setp(legend.get_texts())
    # make label front bigger
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    # save high quality image
    plt.savefig('loss_curve.jpg', dpi=300)
    plt.show()