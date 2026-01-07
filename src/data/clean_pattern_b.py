import os, shutil, glob
from PIL import Image

RAW = "/dataset"                    # mounted dataset
DEST = "../../data/interim"
REAL_DIR = os.path.join(DEST, "real")
FORG_DIR = os.path.join(DEST, "forged")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def ensure_clean_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith(IMG_EXTS):
            out.append(p)
    return sorted(out)

def to_png(src, dst):
    img = Image.open(src).convert("L")
    img.save(dst)

def copy_split(split):
    split_path = os.path.join(RAW, split)
    writer_ids = sorted([
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()
    ])

    for wid in writer_ids:
        gen_dir = os.path.join(split_path, wid)
        forg_dir = os.path.join(split_path, wid + "_forg")

        genuine = list_images(gen_dir)
        forged = list_images(forg_dir)

        for i, src in enumerate(genuine):
            dst = os.path.join(REAL_DIR, "{}_{}_{}.png".format(wid, split, i))
            to_png(src, dst)

        for i, src in enumerate(forged):
            dst = os.path.join(FORG_DIR, "{}_{}_{}.png".format(wid, split, i))
            to_png(src, dst)

if __name__ == "__main__":
    ensure_clean_dir(REAL_DIR)
    ensure_clean_dir(FORG_DIR)

    copy_split("train")
    copy_split("test")

    print("Done.")
    print("Real images:", len(glob.glob(os.path.join(REAL_DIR, "*.png"))))
    print("Forged images:", len(glob.glob(os.path.join(FORG_DIR, "*.png"))))
