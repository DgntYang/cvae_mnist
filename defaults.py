from yacs.config import CfgNode as CN

_C = CN()

# Data
_C.data = CN()
_C.data.data_name = ''
_C.data.data_folder = ''
_C.data.size = 0
_C.data.n_channel = 1

# Model
_C.model = CN()
_C.model.model_name = ''
_C.model.latent_size = 32

# Dataloader
_C.data_loader = CN()
_C.data_loader.batch_size = 64
_C.data_loader.num_workers = 0
_C.data_loader.pin_memory = True

# Solver
_C.solver = CN()
_C.solver.num_epochs = 50
_C.solver.optimizer = ''
_C.solver.lr = 1e-3
_C.solver.weight_decay = 1e-2

#Mode
_C.mode = ''

#Checkpoint
_C.checkpoint = CN()
_C.checkpoint.savepath = ''
_C.checkpoint.loadpath = ''

#Generated pics
_C.generated_path = ''

def get_cfg():
    return _C.clone()


