import torch.nn as nn


class Hook():
    def __init__(self,module,backward=False):
        if backward == False:
            self.hook = module.register_foward_hook(self.hook_fn)
    def hook_fn(self,module,input,output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()