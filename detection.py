import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import matplotlib
matplotlib.use('AGG')
import numpy as np
#from sklearn.metrics import roc_auc_score

plt.style.use('ggplot')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--txt_1', type=str, default='100')
parser.add_argument('--txt_2', type=str, default='100')
parser.add_argument('--label_1', type=str, default='100')
parser.add_argument('--label_2', type=str, default='100')
parser.add_argument('--threshold', type=float, default=1e-4)
parser.add_argument('--save_name', type=str, default='100')

args = parser.parse_args()

list_1 = []
with open(args.txt_1) as f1:
    for line in f1:
        list_1.append(float(line))

list_2 = []
with open(args.txt_2) as f2:
    for line in f2:
        list_2.append(float(line))

print(list_1)
print(list_2)


threshold = args.threshold


fp=0
fn=0
for i in list_1:
    if i>threshold:
        fn = fn + 1
tp = len(list_1) - fn
for j in list_2:
    if j<threshold:
        fp = fp + 1
tn = len(list_2) - fp

print("tp,fp,fn,tn:",tp,fp,fn,tn)


bins_1 = 30
bins_2 = 150
plt.hist(list_1, bins_1, weights=np.ones(len(list_1)) / len(list_1), alpha=0.5, label=args.label_1)
plt.hist(list_2, bins_2, weights=np.ones(len(list_2)) / len(list_2), alpha=0.5, label=args.label_2)

plt.legend(loc='best', fontsize = 14)
plt.tick_params(labelsize=22)
plt.xlabel('Reconstructed MSE Loss', fontsize = 22)
plt.ylabel('Percentage', fontsize = 22)

plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
plt.savefig(args.save_name, dpi=600)
plt.close('all')
