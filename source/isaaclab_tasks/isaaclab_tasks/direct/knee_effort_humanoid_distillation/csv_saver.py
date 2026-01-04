import csv
import os
from datetime import datetime
import numpy as np


class CSVSaver:
    def __init__(self, filename=None, directory="./data", fieldnames=None):
        """
        初始化 CSV 保存器

        Args:
            filename: 文件名，默认使用时间戳生成
            directory: 保存目录，默认为 "data"
            fieldnames: 列名列表（用于数组数据），如 ["x", "y", "z"]
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if filename is None:
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.filepath = os.path.join(directory, filename)
        self.fieldnames = fieldnames
        self.header_written = False

    def save_row(self, data):
        """
        保存单行数据

        Args:
            data: 字典、numpy数组或列表
                  字典格式: {"time": 1.0, "x": 0.5, "y": 0.3}
                  数组格式: [1.0, 0.5, 0.3] 或 np.array([1.0, 0.5, 0.3])
        """
        file_exists = os.path.exists(self.filepath)

        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            # 如果是 numpy 数组或列表
            if isinstance(data, (np.ndarray, list, tuple)):
                writer = csv.writer(f)

                # 写入表头（如果有 fieldnames 且还没写过）
                if self.fieldnames and (not file_exists or not self.header_written):
                    writer.writerow(self.fieldnames)
                    self.header_written = True

                # 写入数据
                if isinstance(data, np.ndarray):
                    writer.writerow(data.tolist())
                else:
                    writer.writerow(data)

            # 如果是字典
            elif isinstance(data, dict):
                writer = csv.DictWriter(f, fieldnames=data.keys())

                if not file_exists or not self.header_written:
                    writer.writeheader()
                    self.header_written = True

                writer.writerow(data)

    def save_rows(self, data_list):
        """
        保存多行数据

        Args:
            data_list: 字典列表或二维数组
        """
        if not data_list:
            return

        file_exists = os.path.exists(self.filepath)

        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            # 判断数据类型
            first_item = data_list[0] if not isinstance(data_list, np.ndarray) else data_list[0]

            # 如果是二维数组或列表的列表
            if isinstance(data_list, np.ndarray) or isinstance(first_item, (np.ndarray, list, tuple)):
                writer = csv.writer(f)

                if self.fieldnames and (not file_exists or not self.header_written):
                    writer.writerow(self.fieldnames)
                    self.header_written = True

                for row in data_list:
                    if isinstance(row, np.ndarray):
                        writer.writerow(row.tolist())
                    else:
                        writer.writerow(row)

            # 如果是字典列表
            elif isinstance(first_item, dict):
                writer = csv.DictWriter(f, fieldnames=first_item.keys())

                if not file_exists or not self.header_written:
                    writer.writeheader()
                    self.header_written = True

                writer.writerows(data_list)

    def get_filepath(self):
        """返回当前文件路径"""
        return self.filepath


def save_to_csv(data, filename, directory="data"):
    """
    简单的一次性保存函数

    Args:
        data: 字典列表或二维数组
        filename: 文件名
        directory: 保存目录
    """
    if not data:
        return

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # 判断是字典列表还是数组
        if isinstance(data[0], dict):
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(f)
            for row in data:
                if isinstance(row, np.ndarray):
                    writer.writerow(row.tolist())
                else:
                    writer.writerow(row)

    print(f"数据已保存到: {filepath}")
    return filepath


# 测试代码
if __name__ == "__main__":
    # 测试1: 字典格式
    print("=== 测试1: 字典格式 ===")
    saver1 = CSVSaver(filename="test_dict.csv")
    saver1.save_row({"time": 1.0, "x": 0.5, "y": 0.3})
    saver1.save_row({"time": 2.0, "x": 0.6, "y": 0.4})
    print(f"文件保存在: {saver1.get_filepath()}\n")

    # 测试2: 数组格式（带列名）
    print("=== 测试2: 数组格式（带列名）===")
    fieldnames = [f"qpos_{i}" for i in range(5)] + [f"qvel_{i}" for i in range(3)]
    saver2 = CSVSaver(filename="test_array.csv", fieldnames=fieldnames)

    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2])
    saver2.save_row(data)
    saver2.save_row([0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 1.2, 1.3])
    print(f"文件保存在: {saver2.get_filepath()}\n")

    # 测试3: 数组格式（不带列名）
    print("=== 测试3: 数组格式（不带列名）===")
    saver3 = CSVSaver(filename="test_array_noheader.csv")
    saver3.save_row(np.array([1, 2, 3, 4, 5]))
    saver3.save_row([6, 7, 8, 9, 10])
    print(f"文件保存在: {saver3.get_filepath()}\n")