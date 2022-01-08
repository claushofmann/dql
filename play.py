import numpy as np
import matplotlib.pyplot as plt
from env.env import BreakoutEnv
from env.renderer import BreakoutRenderer
import time

instructions = """
Player A:       Player B:
  'e'      up     'i'
  'd'     down    'k'

press 't' -- close these instructions
            (animation will be much faster)
press 'a' -- add a puck
press 'A' -- remove a puck
press '1' -- slow down all pucks
press '2' -- speed up all pucks
press '3' -- slow down distractors
press '4' -- speed up distractors
press ' ' -- reset the first puck
press 'n' -- toggle distractors on/off
press 'g' -- toggle the game on/off

  """


class Game:
    def __init__(self, ax):
        self.env = BreakoutEnv(5)
        self.renderer = BreakoutRenderer(self.env)
        self.done = False
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.last_key_pressed = None
        self.cnt = 0
        self.background = None

    def draw(self, event):
        [p.remove() for p in reversed(ax.patches)]
        if self.background is None:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # restore the clean slate background
        self.canvas.restore_region(self.background)
        if self.last_key_pressed is None:
            state, _, done = self.env.step(1)
        if self.last_key_pressed == 'a':
            state, _, done = self.env.step(0)
        if self.last_key_pressed == 'd':
            state, _, done = self.env.step(2)
        self.renderer.render(state, self.ax)
        self.last_key_pressed = None
        # just redraw the axes rectangle
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()
        self.cnt += 1
        #plt.show()
        self.ax.figure.canvas.draw_idle()
        if done:
            plt.close()
        return True

    def on_key_press(self, event):
        if event.key in {'a', 'd'}:
            self.last_key_pressed = event.key




fig, ax = plt.subplots()
canvas = ax.figure.canvas
animation = Game(ax)

# disable the default key bindings
if fig.canvas.manager.key_press_handler_id is not None:
    canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)


# reset the blitting background on redraw
def on_redraw(event):
    animation.background = None


# bootstrap after the first draw
def start_anim(event):
    canvas.mpl_disconnect(start_anim.cid)

    def local_draw():
        if animation.ax.get_renderer_cache():
            animation.draw(None)
    start_anim.timer.add_callback(local_draw)
    start_anim.timer.start()
    canvas.mpl_connect('draw_event', on_redraw)


start_anim.cid = canvas.mpl_connect('draw_event', start_anim)
start_anim.timer = animation.canvas.new_timer(interval=1)

tstart = time.time()

plt.show()
print('FPS: %f' % (animation.cnt/(time.time() - tstart)))
