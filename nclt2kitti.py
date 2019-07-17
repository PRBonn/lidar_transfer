#!/usr/bin/env python3

import sys
import struct
import numpy as np
import os
from pathlib import Path

def progressbar(name, value, endvalue, bar_length=50):
  percent = float(value) / endvalue
  arrow = '-' * int(round(percent * bar_length) - 1) + '|'
  spaces = ' ' * (bar_length - len(arrow))
  sys.stdout.write("\r")
  sys.stdout.write(" " * 80)
  sys.stdout.write("\r{0} {1}: [{2}] {3}%".format(
                   name, value, arrow + spaces, int(round(percent * 100))))
  sys.stdout.flush()

def convert(x_s, y_s, z_s):
    # - Data was recorded as a 2 bytes.
    # - Distances are scaled to an integer between 0 and 40 000 by adding 100 m
    #   to each distance and discretizing the result at 5 mm
    scaling = 0.005 # 5 mm
    offset = -100.0
    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    # In NCTL velodyne mounted upside down
    z *= -1
    return x, y, z

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./nclt2kitty.py <input_dir> <output_dir> <scene_name>")
        exit(1)
    
    # read parameter
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    scene_name = str(sys.argv[3])

    scene_dir = os.path.join(output_dir, "sequences", scene_name) 

    scan_files = []
    for dirName, subdirList, fileList in os.walk(input_dir):
        for fname in sorted(fileList):
            if fname.endswith(".bin"):
                scan_files.append(os.path.join(dirName, fname))

    print("-- Creating directories.")
    # TODO check if all folder exists
    velodyne_dir = os.path.join(scene_dir, "velodyne")
    if not os.path.exists(velodyne_dir):
        Path(velodyne_dir).mkdir(parents=True)

    progress = 0
    for idx, file in enumerate(scan_files):
        progressbar("Scan", idx, len(scan_files))

        f_bin = open(file, "rb")
        out_file = open(os.path.join(scene_dir, "velodyne", str(idx).zfill(6)+".bin"), "wb")
        while True:
            x_str = f_bin.read(2)
            if x_str == '': # eof
                break

            try:
                x = struct.unpack('<H', x_str)[0]
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]
                l = struct.unpack('B', f_bin.read(1))[0]
            except Exception as e:
                # print("Exception while unpacking ...", e)
                break

            x, y, z = convert(x, y, z)
            # s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
            byte_values = struct.pack("ffff", x, y, z, i)
            out_file.write(byte_values)
        f_bin.close()
        out_file.close()
print()
