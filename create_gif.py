import re
import imageio.v3 as iio
import glob


def extract_epoch(filename):
    # Extract the number from something like "voronoi_epoch_42.png"
    match = re.search(r"voronoi_epoch_(\d+)\.png", filename)
    return int(match.group(1)) if match else -1


files = sorted(glob.glob("figures/5_classes/voronoi_epoch_*.png"), key=extract_epoch)
images = [iio.imread(f) for f in files]
iio.imwrite("gifs/voronoi_evolution_5_classes.gif", images, fps=1)
