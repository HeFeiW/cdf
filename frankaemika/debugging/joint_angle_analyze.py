# Joint Angle Analysis for LeapHand data
'''
数据格式说明：
保存路径：<save_dir>/<robot>/<method>/finger_<i>/data.npy
数据结构为 Python dict（numpy 的 object array 格式）：
data = {
    point_idx: {           # int，task space 格点索引 (0 ~ N_t^3 - 1)
        'x':   np.ndarray, # shape (3,)，task space 格点坐标（米）
        'q':   np.ndarray, # shape (M, dof)，M 个满足接触条件的关节角配置
        'idx': np.ndarray, # shape (M,)，每个配置的最近接触 link 索引
    },
    ...
}
'''

import numpy as np
import matplotlib.pyplot as plt

def analyze_joint_angles(data_path, batch_size=1000):
    '''
    分析 LeapHand 数据中的关节角分布和接触 link 分布.(由于数据量较大，进行分批次的增量式处理)
    Args:
        data_path (str): 数据文件路径，应该是一个 .npy 文件，包含上述格式的数据。
        batch_size (int): 每次处理的数据点数量，默认为 1000。
    '''
    try:
        data = np.load(data_path, allow_pickle=True).item()  # 加载数据
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    point_indices = list(data.keys())
    if len(point_indices) == 0:
        print("数据为空，无法分析。")
        return

    dof = data[point_indices[0]]['q'].shape[1]  # 关节自由度

    # 固定关节角统计范围与 bin，按关节分别累计计数
    q_bin_edges = np.arange(-1.5, 1.5 + 0.1, 0.1)
    q_hist_counts = np.zeros((dof, len(q_bin_edges) - 1), dtype=np.int64)
    link_hist_counts = np.zeros(0, dtype=np.int64)
    
    for start_idx in range(0, len(point_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(point_indices))
        batch_indices = point_indices[start_idx:end_idx]
        print(f"Processing points {start_idx} to {end_idx}...")
        for idx in batch_indices:
            point_data = data[idx]
            q_configs = point_data['q']  # shape (M, dof)
            link_indices = point_data['idx']  # shape (M,)

            if q_configs.size > 0:
                for joint_idx in range(dof):
                    joint_counts, _ = np.histogram(q_configs[:, joint_idx], bins=q_bin_edges)
                    q_hist_counts[joint_idx] += joint_counts

            if link_indices.size > 0:
                link_indices = link_indices.astype(np.int64)
                max_link = int(np.max(link_indices))
                if max_link >= len(link_hist_counts):
                    new_counts = np.zeros(max_link + 1, dtype=np.int64)
                    new_counts[:len(link_hist_counts)] = link_hist_counts
                    link_hist_counts = new_counts
                link_hist_counts += np.bincount(link_indices, minlength=len(link_hist_counts))

    # 可视化关节角分布
    fig_width = max(12, 3 * dof)
    plt.figure(figsize=(fig_width, 4))
    plt.subplots_adjust(wspace=0.4)

    q_bin_centers = (q_bin_edges[:-1] + q_bin_edges[1:]) / 2
    q_bin_width = q_bin_edges[1] - q_bin_edges[0]

    for joint_idx in range(dof):
        plt.subplot(1, dof, joint_idx + 1)
        plt.bar(q_bin_centers, q_hist_counts[joint_idx], width=q_bin_width * 0.9, color='blue')
        plt.xlabel('Joint Angle (radians)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.title(f'Joint {joint_idx} Angle Distribution')

    plt.savefig('joint_angle_distribution.png')
    plt.close()

    # 可视化接触 link 分布
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(link_hist_counts)), link_hist_counts, width=0.8, color='orange')
    plt.title('Contact Link Index Distribution')
    plt.xlabel('Contact Link Index')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('contact_link_index_distribution.png')
    plt.close()
    print("分析完成，结果已保存为 joint_angle_distribution.png 和 contact_link_index_distribution.png")
if __name__ == "__main__":
    data_path = 'data/leaphand/data_with_base_dof_0.npy'  # 替换为实际数据路径
    analyze_joint_angles(data_path, batch_size=1000)