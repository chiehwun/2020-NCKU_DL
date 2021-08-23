import numpy as np
from matplotlib import pyplot as plt
global ax

fig, ax = plt.subplots()
ax.plot(np.random.rand(10))
title = plt.title('123123')
# text = plt.text(0, 0, 'PPP', fontsize=16)

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    title.set_text((event.xdata, event.ydata))
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
