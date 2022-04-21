#!/usr/bin/env python
import sys
from telnetlib import theNULL
from multiprocessing import Event
import numpy as np
import airsim
import time
import cv2
import matplotlib.pyplot as plt
import argparse
import sys, signal
import pandas as pd
import pickle
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from numba.np.ufunc import parallel
from types import SimpleNamespace
from numba import njit, prange, set_num_threads
from dvs_msgs.msg import Event, EventArray 

parser = argparse.ArgumentParser(description="Simulate event data from AirSim")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--height", type=int, default=144)
parser.add_argument("--width", type=int, default=256)


eventmessage = Event() 
eventarraymessage = EventArray ()




EVENT_TYPE = np.dtype(
    [("x", "u2"), ("y", "u2"), ("timestamp", "f8"), ("polarity", "b")], align=True
)

TOL = 0.5
MINIMUM_CONTRAST_THRESHOLD = 0.01

CONFIG = SimpleNamespace(
    **{
        "contrast_thresholds": (0.01, 0.01),
        "sigma_contrast_thresholds": (0.0, 0.0),
        "refractory_period_ns": 1000,
        "max_events_per_frame": 200000,
    }
)


@njit(parallel=True)
def esim(
    x_end,
    current_image,
    previous_image,
    delta_time,
    crossings,
    last_time,
    output_events,
    spikes,
    refractory_period_ns,
    max_events_per_frame,
    n_pix_row,
):
    count = 0
    max_spikes = int(delta_time / (refractory_period_ns * 1e-3))
    for x in prange(x_end):
        itdt = np.log(current_image[x])
        it = np.log(previous_image[x])
        deltaL = itdt - it

        if np.abs(deltaL) < TOL:
            continue

        pol = np.sign(deltaL)

        cross_update = pol * TOL
        crossings[x] = np.log(crossings[x]) + cross_update

        lb = crossings[x] - it
        ub = crossings[x] - itdt

        pos_check = lb > 0 and (pol == 1) and ub < 0
        neg_check = lb < 0 and (pol == -1) and ub > 0

        spike_nums = (itdt - crossings[x]) / TOL
        cross_check = pos_check + neg_check
        spike_nums = np.abs(int(spike_nums * cross_check))

        crossings[x] = itdt - cross_update
        if spike_nums > 0:
            spikes[x] = pol

        spike_nums = max_spikes if spike_nums > max_spikes else spike_nums

        current_time = last_time
        for i in range(spike_nums):
            output_events[count].x = x % n_pix_row
            output_events[count].y = x // n_pix_row
            ##output_events[count].timestamp = rospy.Time.now()
            output_events[count].timestamp = np.round(current_time * 1e-6, 6)

            output_events[count].polarity = 1 if pol > 0 else -1

            count += 1
            current_time += (delta_time) / spike_nums

            if count == max_events_per_frame:
                return count

    return count


class EventSimulator:
    def __init__(self, W, H, first_image=None, first_time=None, config=CONFIG):
        self.H = H
        self.W = W
        self.config = config
        self.last_image = None
        if first_image is not None:
            assert first_time is not None
            self.init(first_image, first_time)

        self.npix = H * W

    def init(self, first_image, first_time):
        print("Initialized event camera simulator with sensor size:", first_image.shape)

        self.resolution = first_image.shape  # The resolution of the image

        # We ignore the 2D nature of the problem as it is not relevant here
        # It makes multi-core processing more straightforward
        first_image = first_image.reshape(-1)

        # Allocations
        self.last_image = first_image.copy()
        self.current_image = first_image.copy()

        self.last_time = first_time

        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.event_count = 0
        self.spikes = np.zeros((self.npix))

    def image_callback(self, new_image, new_time):
        if self.last_image is None:
            self.init(new_image, new_time)
            return None, None

        assert new_time > 0
        assert new_image.shape == self.resolution
        new_image = new_image.reshape(-1)  # Free operation

        np.copyto(self.current_image, new_image)

        delta_time = new_time - self.last_time

        config = self.config
        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.spikes = np.zeros((self.npix))

        self.crossings = self.last_image.copy()
        self.event_count = esim(
            self.current_image.size,
            self.current_image,
            self.last_image,
            delta_time,
            self.crossings,
            self.last_time,
            self.output_events,
            self.spikes,
            config.refractory_period_ns,
            config.max_events_per_frame,
            self.W,
        )
        if self.event_count == None: 
            self.event_count = 0;
        np.copyto(self.last_image, self.current_image)
        self.last_time = new_time

        result = self.output_events[: self.event_count]
        result.sort(order=["timestamp"], axis=0)

        return self.spikes, result











class AirSimEventGen:
    def __init__(self, W, H, save=False, debug=False):
        self.ev_sim = EventSimulator(W, H)
        self.H = H
        self.W = W

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False
        )

        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        self.init = True
        self.start_ts = None

        self.rgb_image_shape = [H, W, 3]
        self.debug = debug
        self.save = save

        self.event_file = open("events.pkl", "ab")
        self.event_fmt = "%1.7f", "%d", "%d", "%d"

        if debug:
            self.fig, self.ax = plt.subplots(1, 1)

    def visualize_events(self, event_img):
        event_img = self.convert_event_img_rgb(event_img)
        self.ax.cla()
        self.ax.imshow(event_img, cmap="viridis")
        plt.draw()
        plt.pause(0.01)

    def convert_event_img_rgb(self, image):
        image = image.reshape(self.H, self.W)
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out[:, :, 0] = np.clip(image, 0, 1) * 255
        out[:, :, 2] = np.clip(image, -1, 0) * -255

        return out

    def _stop_event_gen(self, signal, frame):
        print("\nCtrl+C received. Stopping event sim...")
        self.event_file.close()
        sys.exit(0)


if __name__ == "__main__":
    args = parser.parse_args()

    event_generator = AirSimEventGen(args.width, args.height, save=args.save, debug=args.debug)
    i = 0
    start_time = 0
    t_start = time.time()

    signal.signal(signal.SIGINT, event_generator._stop_event_gen)



    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    rospy.init_node('EventPublisher', anonymous=True)
    pub = rospy.Publisher('Events', numpy_msg(Event), queue_size=100) #pub is going to be in charge of separate events
    arraypub = rospy.Publisher('EventArray', EventArray, queue_size=100)
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    while True:
        image_request = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)

        response = event_generator.client.simGetImages([event_generator.image_request])
        while response[0].height == 0 or response[0].width == 0:
            response = event_generator.client.simGetImages(
                [event_generator.image_request]
            )

        ts = time.time_ns()

        if event_generator.init:
            event_generator.start_ts = ts
            event_generator.init = False

        img = np.reshape(
            np.frombuffer(response[0].image_data_uint8, dtype=np.uint8),
            event_generator.rgb_image_shape,
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Add small number to avoid issues with log(I)
        img = cv2.add(img, 0.001)

        ts = time.time_ns()
        ts_delta = (ts - event_generator.start_ts) * 1e-3

        # Event sim keeps track of previous image automatically
        event_img, events = event_generator.ev_sim.image_callback(img, ts_delta)
        # ROS USAGE HERE
        

        if events != None:
            for i in events:
                eventmessage.x = i[0].astype('uint8')
                eventmessage.y = i[1].astype('uint8')
                eventmessage.ts = rospy.Duration.from_sec(i[2])
                if i[3]==1:
                    eventmessage.polarity = True
                else:
                    eventmessage.polarity = False

                eventarraymessage.events.append(eventmessage)
                #rospy.loginfo(eventmessage)
                #pub.publish(eventmessage)

            rospy.loginfo(eventarraymessage)
            arraypub.publish(eventarraymessage)
            eventarraymessage.events.clear()