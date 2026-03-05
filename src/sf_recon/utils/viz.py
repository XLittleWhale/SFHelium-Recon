import matplotlib.pyplot as plt
import numpy as np

def streamline_animation(u_data, v_data, X, Y, markers_positions, Lx, Ly, interval=200):
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        ax.streamplot(X, Y, u_data[frame], v_data[frame], color='cornflowerblue')
        px = markers_positions[frame,:,0]
        py = markers_positions[frame,:,1]
        ax.scatter(px, py, color='orange')
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
        return ax,
    ani = animation.FuncAnimation(fig, update, frames=len(u_data), interval=interval)
    return ani
