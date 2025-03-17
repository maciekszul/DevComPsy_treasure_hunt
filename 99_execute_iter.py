import sys
import subprocess as sp

try:
    path = str(sys.argv[1])
except:
    raise IndexError("no file path")

try:
    range_of_files = int(sys.argv[2])
except:
    raise IndexError("no range of files")

try:
    json_file = sys.argv[3]
except:
    json_file = "settings.json"

for index in range(range_of_files):
    sp.call([
        "python",
        path,
        str(index),
        json_file
    ])