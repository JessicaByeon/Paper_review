# https://github.com/godeastone/GAN-torch/blob/main/models/GAN.py
# 참고 https://ddongwon.tistory.com/124

import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image


# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "GAN_results"

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024


# Device setting --- cuda gpu 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))


# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Dataset transform setting
# transform.Compose 데이터 전처리, 타입/데이터 argument 하기 위한 함수 내장
# 여러 단계로 변환해야 하는 경우, Compose를 통해 여러 단계를 묶을 수 있다. (transforms 함수들을 compose로 묶어서 한번에 처리)
# transforms.ToTensor() : 데이터 타입을 Tensor로 변경
# transform.Normalize() : 데이터 정규화, 이미지의 경우 픽셀 값 하나는 0~255 사이의 값을 가짐
# 하지만 totensor()로 타입변경 시 0~1 사이의 값으로 바뀜
# transforms.Normalize 를 이용하여 -1~1 사이 값으로 정규화

# Normalize -- 크게 두가지 연산으로 나눠짐.
# scaling : 데이터의 scale을 줄여줌.
# centering : 데이터의 중심을 원점으로 맞춰주는 것.
# ToTensor()를 해주면 scaling, Normalize를 해주면 centering + rescaling.
# 정확하게 하기 위해선 학습 데이터로부터 각 픽셀별로 평균을 구하거나 채널별로 평균을 구해서 centering 해야함.


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

# normalize는 image = (image - mean) / std 로 정규화를 해줌.
# transforms.Normalize(mean, std, inplace=False)
# ()안의 수가 채널의 수, 각 채널마다 mean(평균)과 std(표준편차)를 적용

# CNN 모델에서는 transforms를 해주는데, 원하는 형태의 데이터로 바꿔주기 위해서!
# 보통은 transforms.Normalize((mean_1, mean_2, mean_3), (std_1, std_2, std_3)) 이렇게 채널별로 mean, std 값을 할당

# MNIST dataset setting
MNIST_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transform,
                                           download=True)
# root : 경로 지정
# train : train or test 데이터를 받아옴.
# transorm : 사전에 설정해 놓은 데이터 처리 형태
# download : 데이터 셋 다운로드 여부


# Data loader
data_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
# dataset : 불러올 데이터 셋
# batch_size : batch 단위 만큼 데이터를 뽑아옴.
# shuffle : shuffle 여부


# Declares discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(img_size, hidden_size3)
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


# Declares generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x


# Initialize generator/Discriminator
discriminator = Discriminator()
generator = Generator()

# Device setting --- GPU 사용
discriminator = discriminator.to(device)
generator = generator.to(device)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)


"""
Training part
"""
for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        # make ground truth (labels) -> 1 for real, 0 for fake
        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        # reshape real images from MNIST dataset
        real_images = images.reshape(batch_size, -1).to(device)

        # +---------------------+
        # |   train Generator   |
        # +---------------------+

        # Initialize grad
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        # torch.randn : 평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 생성한 텐서를 반환
        
        
        
        # Compare result of discriminator with fake images & real labels
        # If generator deceives discriminator, g_loss will decrease
        g_loss = criterion(discriminator(fake_images), real_label)

        # Train generator with backpropagation
        g_loss.backward()
        g_optimizer.step()

        # +---------------------+
        # | train Discriminator |
        # +---------------------+

        # Initialize grad
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        # Calculate fake & real loss with generated images above & real images
        fake_loss = criterion(discriminator(fake_images), fake_label)
        real_loss = criterion(discriminator(real_images), real_label)
        d_loss = (fake_loss + real_loss) / 2

        # Train discriminator with backpropagation
        # In this part, we don't train generator
        d_loss.backward()
        d_optimizer.step()

        d_performance = discriminator(real_images).mean()
        g_performance = discriminator(fake_images).mean()

        if (i + 1) % 150 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                  .format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))

    # print discriminator & generator's performance
    print(" Epock {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"
          .format(epoch, d_performance, g_performance))

    # Save fake images in each epoch
    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, 'GAN_fake_samples{}.png'.format(epoch + 1)))
    
    