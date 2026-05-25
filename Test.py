from pathlib import Path
from helper import get_df_all
import warnings
warnings.filterwarnings("ignore")             # 忽略无关警告
from typing import Optional, List



working_dir = Path('.')
DATA_PATH = Path("./Data")
save_model_path = working_dir / 'Model'
DE_path = DATA_PATH / '12k_DE'

for path in [DATA_PATH, save_model_path]:
    if not path.exists():
        path.mkdir(parents=True)

# 加载数据
df_all = get_df_all(DE_path, segment_length=500, normalize=True)


feature_cols = df_all.columns[2:]  # 仅保留特征列名
features = df_all[feature_cols]  # 特征 DataFrame
target = 'label'
labels = df_all[target].tolist()



# 生成 CWT 图像
# # CWT 参数设置
# sampling_period = 1.0 / 12000
# totalscal = 128
# wavename = 'cmor1-1'
# fc = pywt.central_frequency(wavename)
# cparam = 2 * fc * totalscal
# scales = cparam / np.arange(totalscal, 0, -1)
#
# # 创建保存图片的文件夹
# image_save_dir = working_dir / 'CWT_Images'
# if not image_save_dir.exists():
#     image_save_dir.mkdir(parents=True)
#
# features_array = features.values
# M = len(features_array)
#
# for i in tqdm(range(M), desc="正在生成时频谱图片"):
#     signal_data = features_array[i]
#
#     # 执行 CWT
#     coeffs, _ = pywt.cwt(signal_data, scales, wavename, sampling_period)
#     amp = np.abs(coeffs)
#
#     # 构造文件路径
#     img_path = image_save_dir / f"sample_{i:04d}.png"
#
#     # 保存为图像（低频在下，与常规频谱图一致）
#     plt.imsave(str(img_path), amp, cmap='jet', origin='lower')
#
# print(f"\n全部图片生成完毕！已保存至文件夹: {image_save_dir}")





def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

import os
from PIL import Image
from torch.utils.data import Dataset











# 定义数据集类
class CWTDataset(Dataset):
    """读取单个文件夹下按顺序命名的时频谱图像及其标签"""
    def __init__(self, image_dir, labels, transform=None):
        """
        Args:
            image_dir (str): 存放所有 .png 图像的文件夹路径
            labels (list or array): 与图像顺序一一对应的标签列表（长度必须等于图像数量）
            transform: 图像变换（albumentations / torchvision.transforms）
        """
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

        # 获取所有 .png 图像路径，并按文件名排序（保证与 labels 的顺序一致）
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith('.png')
        ])

        # 安全检查：图像数量必须与标签数量相同
        if len(self.image_paths) != len(self.labels):
            raise ValueError(
                f"图像数量 ({len(self.image_paths)}) 与标签数量 ({len(self.labels)}) 不一致！"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图像并转为 RGB（生成的时频谱图为彩色，无需转换通道）
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label







from torchvision import transforms
# 数据样式
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



# 数据划分
import torch
from torch.utils.data import DataLoader
dataset = CWTDataset(image_dir='CWT_Images',  labels=labels, transform=train_transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False)






# 模型调用和修改(ResNet50+瓶颈层)
import torch.nn as nn
from torchvision.models import resnet50
model = resnet50(pretrained=True)
model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 102),
)
# 损失函数
criterion = nn.CrossEntropyLoss()





# 模型训练
device = torch.device("cuda")
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)




# 开始训练
for epoch in range(20):
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_data_loader),
        [losses, acc],
        prefix="Epoch: [{}]".format(epoch)
    )
    for i, data in enumerate(train_data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        cls_acc = accuracy(outputs, labels)[0]
        losses.update(loss.item(), images.size(0))
        acc.update(cls_acc.item(), images.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            progress.display(i)

    # 模型测试
    with torch.no_grad():
        model.eval()
        test_losses = AverageMeter('Test_Loss', ':.4f')
        test_acc = AverageMeter('Test_Acc', ':6.2f')
        progress = ProgressMeter(
            len(test_data_loader),
            [test_losses, test_acc],
            prefix='Test: '
        )
        for i, data in enumerate(test_data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)[0]
            test_losses.update(loss.item(), images.size(0))
            test_acc.update(acc.item(), images.size(0))
            if i % 100 == 0:
                progress.display(i)








