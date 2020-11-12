import matplotlib.pyplot as plt
import numpy as np


labels = ['gaussian 1%', 'gaussian 10%', 'gaussian 20%']
dct_means = [0.03, 0.21, 0.27]
qk_means = [0.02, 0.11, 0.16]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dct_means, width, label='DCT')
rects2 = ax.bar(x + width/2, qk_means, width, label='q-Krawtchouk')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('BER')
ax.set_title('Puntuaciones por grupo y transformada')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
