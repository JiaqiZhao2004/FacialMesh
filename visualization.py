from matplotlib import animation

from data import roy_turn_head
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(roy_turn_head)

# Index(['chin center left', 'chin center right', 'left eye', 'right eye',
#        'left sun x', 'left sun y', 'right sun x', 'right sun y', 'mouth'],
#       dtype='object')




fig, ax = plt.subplots()


def update(frame, df=df):
    # for each frame, update the data stored on each artist.
    df = df[:frame]
    # update the scatter plot:
    a = ax.scatter(df['right sun x'], df['left sun x'], c="red")
    b = ax.scatter(df['right sun y'], df['left sun y'], c="orange")
    c = ax.scatter(df['right sun x'], df['chin center left'], c="yellow")
    d = ax.scatter(df['right sun y'], df['chin center right'], c="green")
    e = ax.scatter(df['left sun x'], df['chin center left'], c="blue")
    f = ax.scatter(df['left sun y'], df['chin center right'], c="purple")

    g = plt.scatter(df['chin center left'], df['chin center right'])
    h = plt.scatter(df['left eye'], df['right eye'])  # -0.08 ~ 0.04
    i = plt.scatter(df['left sun x'], df['left sun y'])
    j = plt.scatter(df['right sun x'], df['right sun y'])


    return a, b, c, d, e, f, g, h, i ,j


ani = animation.FuncAnimation(fig=fig, func=update, frames=4041, interval=10)
plt.show()
