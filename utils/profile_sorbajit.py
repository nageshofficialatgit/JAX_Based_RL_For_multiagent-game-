import os

def get_size(start_path):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except:
                        pass
    except:
        pass
    return total_size

import sys
if len(sys.argv) > 1:
    base = sys.argv[1]
else:
    print("Please provide a directory path.")
    sys.exit(1)
sizes = []
for item in os.listdir(base):
    path = os.path.join(base, item)
    if os.path.isdir(path):
        sizes.append((item, get_size(path)))
    else:
        sizes.append((item, os.path.getsize(path)))

sizes.sort(key=lambda x: x[1], reverse=True)
for name, size in sizes:
    if size > 1024 * 1024 * 1024:
        print(f"{size / (1024*1024*1024):.2f} GB - {name}")
    elif size > 1024 * 1024:
        print(f"{size / (1024*1024):.2f} MB - {name}")
    else:
        print(f"{size / 1024:.2f} KB - {name}")
