import os
import shutil
import codecs
import torch
import numpy as np
import matplotlib.pyplot as plt


hashmap = {
        'Mnist':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'cifar10':['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        }


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


PROJECT_ROOT_DIR = './Images'
def image_path(fig_id):
    temp_dir=os.path.join(PROJECT_ROOT_DIR,"test_images")
    return os.path.join(temp_dir, 'test%s'%fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)
    plt.close()

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2])).view(*s)

def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()

def load_data(mode, raw_folder):
    image_file = f"{'train' if mode=='train' else 't10k'}-images-idx3-ubyte"
    data = read_image_file(os.path.join(raw_folder, image_file))

    label_file = f"{'train' if mode=='train' else 't10k'}-labels-idx1-ubyte"
    targets = read_label_file(os.path.join(raw_folder, label_file))

    return data, targets

