import matplotlib.pyplot as plt
import re
import sys

losses = []
with open('nohup.out', 'r') as file:
    for line in file:
        if re.search('Avg. Loss', line):
            loss = float(line.split(' ')[3].split(',')[0])
            losses.append(loss)

losses = losses[0:2] + losses[2::2]

plt.plot(losses, label='train_loss')
plt.show()
plt.savefig(f'./plot_moco_train_loss.jpg')
plt.clf()