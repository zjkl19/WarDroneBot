import os, time
from war_drone.adb_client import ADBClient
from war_drone.logger import Logger

adb = ADBClient()
log = Logger()
img = adb.screencap()
fn = log.dump_img(img, "manual_grab")
print("saved:", fn)
