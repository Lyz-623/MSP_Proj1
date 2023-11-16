import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curve(csv_file_path):
    # 从CSV文件中读取数据
    data = pd.read_csv(csv_file_path)

    # 提取训练步骤和对应的loss列
    steps = data.iloc[:, 0]
    loss_values = data.iloc[:, 1]

    # 绘制loss曲线
    plt.plot(steps, loss_values, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
