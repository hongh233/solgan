# -*- coding: utf-8 -*-




import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from data_loader import get_loader, get_loader_unlabel
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class ResBlkUp(nn.Module):
    def __init__(self, inplane, plane):
        super(ResBlkUp, self).__init__()
        self.learned_shortcut = (inplane != plane)

        self.resdual = nn.Sequential(nn.BatchNorm2d(inplane),
                                     nn.LeakyReLU(0.2, True),
                                     nn.Conv2d(inplane, plane, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(plane),
                                     nn.LeakyReLU(0.2, True),
                                     nn.Conv2d(plane, plane, kernel_size=3, stride=1, padding=1))

        if self.learned_shortcut:
            self.short = nn.Conv2d(inplane, plane, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        resdual = self.resdual(x)
        if self.learned_shortcut:
            shortcut = self.short(x)
        else:
            shortcut = x
        out = resdual + shortcut

        return F.interpolate(out, scale_factor=2)


class ResBlkDown(nn.Module):

    def __init__(self, in_channle, out_channle, stride=2):
        super(ResBlkDown, self).__init__()
        self.learn = stride > 1 or in_channle != out_channle
        self.resdual = nn.Sequential(nn.Conv2d(in_channle, out_channle, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(out_channle),
                                     nn.LeakyReLU(0.2, True),
                                     nn.Conv2d(out_channle, out_channle, kernel_size=3, stride=stride, padding=1),
                                     nn.BatchNorm2d(out_channle)
                                     )
        if self.learn:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channle, out_channle, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channle))

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        down = x
        if self.learn:
            down = self.shortcut(down)
        return self.relu(down + self.resdual(x))


class Generator(nn.Module):

    def __init__(self, input_dim, num_class):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_class, 100)
        self.linear = nn.Linear(input_dim + 100, 256 * 4 * 4)
        self.model5 = ResBlkUp(256, 256)
        self.model4 = ResBlkUp(256, 256)
        self.model3 = ResBlkUp(256, 128)
        self.model2 = ResBlkUp(128, 64)
        self.model1 = ResBlkUp(64, 32)
        self.attn = Self_Attn(128)
        self.out = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z, y):
        embed = self.embed(y)
        x = self.linear(torch.cat([z, embed], dim=1)).view(z.size(0), -1, 4, 4)
        x = self.model5(x)
        x = self.model4(x)
        x = self.model3(x)
        x, attn = self.attn(x)
        x = self.model2(x)
        x = self.model1(x)
        o = self.out(x)
        return o, attn


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.model1 = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1),
                                    nn.LeakyReLU(0.2, True))
        self.model2 = ResBlkDown(32, 64)
        self.model3 = ResBlkDown(64, 128)
        self.model4 = ResBlkDown(128, 256)
        self.model5 = ResBlkDown(256, 256)

        self.attn = Self_Attn(128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(256, num_classes)
        self.dis = nn.Linear(256, 1)

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x, attn = self.attn(x)
        x = self.model4(x)
        x = self.model5(x)
        h = self.pool(x).view(x.size(0), -1)
        return self.cls(h), self.dis(h), attn


def calc_gradient_penalty(netD, x):
    x.requires_grad_(True)
    _, disc_interpolates, _ = netD(x)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=x,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(0),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(x.size(0), -1)
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1)) ** 2).mean() * 10

    return gradient_penalty


batch_size = 128
epochs_gan = 20
epochs_class = 20
lr = 2e-4
input_dim = 100
num_classes = 10

train_data = get_loader('fruit/train/', batch_size=batch_size)
test_data = get_loader('fruit/test/', batch_size=batch_size)


def train_gan(epochs):
    G = Generator(input_dim, num_classes).cuda()
    D = Discriminator(num_classes).cuda()

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.999))

    criterionGAN = nn.Softplus()
    criterionCls = nn.CrossEntropyLoss()

    losses_G = []
    losses_D = []
    losses_cls = []

    G.train()
    D.train()

    for epoch in range(epochs):

        for i, (real, label) in enumerate(train_data):

            real_label = torch.autograd.Variable(torch.ones(real.size(0), 1)).cuda()
            fake_label = torch.autograd.Variable(torch.zeros(real.size(0), 1)).cuda()

            real = real.cuda()
            label = label.cuda()

            noise = torch.randn(real.size(0), input_dim).cuda()
            fake_gen, g_attn = G(noise, label)

            # train generator
            g_cls, g_real, _ = D(fake_gen)
            G_loss_GAN = criterionGAN(-g_real).mean()
            G_loss_cls = criterionCls(g_cls, label)
            loss_G = G_loss_GAN + G_loss_cls
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # train discriminator
            d_real_cls, D_real, d_attn = D(real)
            d_fake_cls, D_fake, _ = D(fake_gen.detach())
            loss_D_real = criterionGAN(-D_real).mean()
            loss_D_fake = criterionGAN(D_fake).mean()
            loss_D_real_cls = criterionCls(d_real_cls, label)
            loss_D_fake_cls = criterionCls(d_fake_cls, label)
            loss_D = loss_D_real + loss_D_fake + loss_D_real_cls + loss_D_fake_cls
            D_loss_GAN = loss_D_real + loss_D_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            if i % 16 == 0:
                opt_D.zero_grad()
                d_loss_gp = calc_gradient_penalty(D, real)
                d_loss_gp.backward()
                opt_D.step()

            # for p in D.parameters():
            #   p.data.clamp_(-0.01, 0.01)

            losses_G.append(G_loss_GAN.item())
            losses_D.append(D_loss_GAN.item())
            losses_cls.append(loss_D_real_cls.item())

        # test
        log = 'epoch:{} G_loss_GAN:{:.3f} G_loss_cls:{:.3f} D_loss_GAN:{:.3f} D_loss_real_cls:{:.3f} D_loss_fake_cls:{:.3f} D_loss_gp:{:.3f}' \
            .format(epoch, G_loss_GAN.item(), G_loss_cls.item(), D_loss_GAN.item(), loss_D_real_cls.item(),
                    loss_D_fake_cls.item(), d_loss_gp.item())
        print(log)
        with open('logs.txt', 'a') as f:
            f.write(log + '\n')

    g_name = 'G_' + '.pth'
    d_name = 'D_' + '.pth'
    torch.save(G.state_dict(), os.path.join('./checkpoint', g_name))
    torch.save(D.state_dict(), os.path.join('./checkpoint', d_name))

    x = np.arange(1, len(losses_G) + 1)

    plot_loss(x, losses_D, 'D_loss', 'loss_D', 'red', 'D_loss.png')
    plot_loss(x, losses_G, 'gan_loss', 'loss_G', 'blue', 'G_loss.png')
    plot_loss(x, losses_cls, 'cls_loss', 'loss_cls', 'orange', 'cls_loss.png')


def train_classifier_with_D(epochs):
    D = Discriminator(num_classes).cuda()
    d_name = 'D_' + '.pth'
    D.load_state_dict(torch.load(os.path.join('checkpoint', d_name)))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.999))

    criterionCls = nn.CrossEntropyLoss()

    Acc = []

    D.train()

    for epoch in range(epochs):

        for real, label in train_data:
            real = real.cuda()
            label = label.cuda()

            cls, _, _ = D(real)
            loss_cls = criterionCls(cls, label)
            opt_D.zero_grad()
            loss_cls.backward()
            opt_D.step()

            prediction = torch.argmax(cls, 1)
            acc = (prediction == label).sum().float()
            acc = (acc / len(label)).cpu()
            Acc.append(acc)

        # test
        print('epoch:{}  loss_cls: {} acc: {}'.format(epoch, loss_cls.item(), acc))

    x = np.arange(1, len(Acc) + 1)

    d_name = 'D_cls' + '.pth'
    torch.save(D.state_dict(), os.path.join('./checkpoint', d_name))

    name = 'acc_' + '.png'
    plot_loss(x, Acc, 'acc', 'acc', 'yellow', name)


def train_classifier_with_D_semi(epochs):
    D = Discriminator(num_classes).cuda()
    d_name = 'D_' + '.pth'
    D.load_state_dict(torch.load(os.path.join('checkpoint', d_name)))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.999))
    criterionCls = nn.CrossEntropyLoss()

    train_data_unlabel = get_loader_unlabel(batch_size=batch_size)

    def cross_entropy(p1, p2):
        p1 = F.softmax(p1, dim=1)
        p2 = F.softmax(p2, dim=1)
        return -(p1 * p2.log()).mean(1).mean()

    Acc = []

    D.train()

    for epoch in range(epochs):

        for (real, label), (fake, fake_aug) in zip(train_data, train_data_unlabel):
            real = real.cuda()
            fake = fake.cuda()
            fake_aug = fake_aug.cuda()
            label = label.cuda()

            cls, _, _ = D(real)
            loss_cls = criterionCls(cls, label)

            cls_fake, _, _ = D(fake)
            pseudo_label = torch.argmax(cls_fake, dim=1).long()
            cls_aug, _, _ = D(fake_aug)
            loss_semi_cls = cross_entropy(cls_fake, cls_fake)

            loss = loss_cls + loss_semi_cls
            opt_D.zero_grad()
            loss.backward()
            opt_D.step()

            prediction = torch.argmax(cls, 1)
            acc = (prediction == label).sum().float()
            acc = (acc / len(label)).cpu()
            Acc.append(acc)

        # test
        print('epoch:{}  loss_cls: {:.3f} loss_semi_cls: {:.3f} acc: {:.3f}'.format(epoch, loss_cls.item(),
                                                                                    loss_semi_cls.item(), acc))

    x = np.arange(1, len(Acc) + 1)

    d_name = 'D_cls' + '.pth'
    torch.save(D.state_dict(), os.path.join('./checkpoint', d_name))

    name = 'acc_' + '.png'
    plot_loss(x, Acc, 'acc', 'acc', 'yellow', name)


def plot_loss(x, y, title, label, color, name):
    # plt.title('title')
    plt.plot(x, y, color=color, label=label)
    plt.legend()
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.savefig(fname=os.path.join('pic/', name))
    plt.show()


def test_gan(num, label):
    G = Generator(input_dim, num_classes).cuda()
    g_name = 'G_' + '.pth'
    G.load_state_dict(torch.load(os.path.join('checkpoint', g_name)))
    G.eval()
    noise = torch.randn(num, input_dim).cuda()
    label = torch.ones(num).fill_(label).long().cuda()
    gen_data, _ = G(noise, label)
    gen_data = gen_data.permute(0, 2, 3, 1)
    gen_data = (((gen_data.detach().cpu().numpy()) * 127.5) + 127.5).astype(np.uint8)
    for i, img in enumerate(gen_data):
        image_pil = Image.fromarray(img)
        image_pil.save(os.path.join('gen', str(i) + '.png'))


def test_acc():
    d_name = 'D_cls' + '.pth'
    model = Discriminator(num_classes).cuda()
    model.load_state_dict(torch.load(os.path.join('checkpoint', d_name)))
    model.eval()
    acc_num = 0
    total = 0
    labels = []
    predicts = []
    for data, label in test_data:
        data = data.cuda()
        label = label.cuda()
        with torch.no_grad():
            cls, _, _ = model(data)
        prediction = torch.argmax(cls, 1)
        acc_num += (prediction == label).sum().float()
        total += len(label)
        labels += label.cpu().numpy().tolist()
        predicts += prediction.cpu().numpy().tolist()

    acc = acc_num / total
    print('test acc is {}'.format(acc))


def generate_gan_heatmap():
    G = Generator(input_dim, num_classes).cuda()
    g_name = 'G_' + '.pth'
    G.load_state_dict(torch.load(os.path.join('checkpoint', g_name)))
    G.eval()
    for i in range(num_classes):
        label = torch.ones(5).fill_(i).long().cuda()
        noise = torch.randn(5, input_dim).cuda()
        imgs, heatmaps = G(noise, label)
        heatmaps = heatmaps.detach().cpu().numpy()
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = (((imgs.detach().cpu().numpy()) * 127.5) + 127.5).astype(np.uint8)
        for j, heatmap in enumerate(heatmaps):
            # heatmap = cv2.resize(heatmap, (128, 128))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
            # cv2.imwrite('heatmap/class{}_{}.png'.format(i,j), heatmap)
            plt.imsave('heatmap/class{}_{}.png'.format(i, j), heatmap)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_gan.png')
    plt.show()


def calc_confusion_matrix():
    d_name = 'D_cls' + '.pth'
    model = Discriminator(num_classes).cuda()
    model.load_state_dict(torch.load(os.path.join('checkpoint', d_name)))
    model.eval()
    conf_matrix = torch.zeros(6, 6)

    def confusion_matrix(preds, labels, conf_matrix):
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix

    for data, label in test_data:
        data = data.cuda()
        label = label.cuda()
        with torch.no_grad():
            cls, _, _ = model(data)
        prediction = torch.argmax(cls, 1)
        conf_matrix = confusion_matrix(prediction, labels=label, conf_matrix=conf_matrix)

    plot_confusion_matrix(conf_matrix.numpy(), classes=[0, 1, 2, 3, 4, 5], normalize=False,
                          title='Normalized confusion matrix')


if __name__ == '__main__':
    flag = 'train_gan'  # [train_gan, train_classifier, test_gan, test_classifier]
    if flag == 'train_gan':
        train_gan(epochs_gan)
    elif flag == 'train_classifier':
        train_classifier_with_D(epochs_class)
    elif flag == 'test_gan':
        test_gan(50, 1)
    elif flag == 'test_classifier':
        test_acc()
    elif flag == 'gen_GAN_heatmap':
        generate_gan_heatmap()
    elif flag == 'plot_confusion_matrix':
        calc_confusion_matrix()


























