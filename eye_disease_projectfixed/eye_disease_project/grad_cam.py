import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.handlers = []
        self.gradients = None
        self.activations = None
        # Find last conv layer if not provided
        if target_layer is None:
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = name
                    break
        self.target_layer = target_layer
        self._register_hooks()

    def _get_module(self, module, access_string):
        """Access module with access string like 'layer1.0.conv1'"""
        for name in access_string.split('.'):
            module = getattr(module, name)
        return module

    def _register_hooks(self):
        module = self._get_module(self.model, self.target_layer)
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.handlers.append(module.register_forward_hook(forward_hook))
        self.handlers.append(module.register_backward_hook(backward_hook))

    def remove_handlers(self):
        for h in self.handlers:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = torch.mean(grads.view(grads.size(0), -1), dim=1)
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w.detach().cpu() * acts[i].detach().cpu()
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        cam = np.uint8(cam.numpy() * 255)
        return cam
