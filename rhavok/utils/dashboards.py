from threading import Thread
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import time

class plt_dashboard:
    def __init__(self, buffer, semaphore, system):
        self.buffer = buffer
        self.semaphore = semaphore
        self.fig = plt.figure(figsize=plt.figaspect(1.6))
        self.dt = 0.001
        self.system = system
        self.thres = 0.002

    def run(self):
        ref_rate = 1/10
        while self.semaphore:
            tic = time.process_time()
            
            data = self.buffer.read_lifo(n=2600)
            nn = len(data)
            ax = self.fig.add_subplot(6, 1, (1, 3), projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], s = 0.7,
                    color=plt.cm.winter(np.linspace(0,1,nn)), label='actor network')
            
            # add Lorenz fixed points
            ax.scatter3D(self.system.fp1[0], self.system.fp1[1], self.system.fp1[2], c='k', s=5)
            ax.scatter3D(self.system.fp2[0], self.system.fp2[1], self.system.fp2[2], c='k', s=5)
            
            #ax.legend()
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.set_title(self.system.name + " System")
            
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
            ax.set_xlim(-30,30)
            ax.set_ylim(-30,30)
            ax.set_zlim(0,50)
            
            ax = self.fig.add_subplot(6, 1, 4) # havok plot
            ax.plot(data[:, 4], c='goldenrod')
            ax.set_ylabel("havok v_r")
            ax.set_xticks([])
            ax.yaxis.tick_right()
            ax.set_ylim(-10*self.thres, 10*self.thres)
            
            ax = self.fig.add_subplot(6, 1, 5) # action plot
            ax.plot(data[:, 5], c='red')
            ax.set_ylabel("action")
            ax.set_xticks([])
            ax.yaxis.tick_right()
            
            ax = self.fig.add_subplot(6, 1, 6) # reward plot
            ax.plot(data[:, 6], c='seagreen')
            ax.set_ylabel("reward")
            ax.set_xticks([])
            ax.yaxis.tick_right()
        
        
            plt.show(block=False)
            
            toc = time.process_time()
            
            plt.pause( ref_rate - min(ref_rate-0.01, toc-tic) ) # overhead?
            
            plt.clf()