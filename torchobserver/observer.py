import torch
import contextlib
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt
import wandb


class BaseObserver:
    def save(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def observe(self, model: torch.nn.Module):
        raise NotImplementedError
    

class Observer(BaseObserver):
    def __init__(self, vocoder=None, sr=None) -> None:
        self._outputs = OrderedDict()
        self._vocoder = vocoder
        self._sr = sr

    def _reset(self):
        self._outputs.clear()

    def _save(self, output, name):
        if 'kwargs' not in output:
            output['kwargs'] = {}

        if output['type'] == 'Mel':
            if self._vocoder is not None:
                output['data'] = self._vocoder(output['data'])
                output['type'] = 'Audio'
            else:
                output['type'] = 'Image'
        
        if output['type'] == 'Audio':
            output['kwargs']['sample_rate'] = self._sr

        if isinstance(output['data'], torch.Tensor):
            output['data'] = output['data'].detach().cpu().numpy()

        wandb.log(
            {f"{output['type']}/{output['name']}/{name}": getattr(wandb, output['type'])(output['data'], **output['kwargs'])}
        )    

    def save(self, key=None):
        for name, output in self._outputs.items():
            if key is not None: name = f"{name}_{key}"
            if isinstance(output, list):
                for out in output:
                    self._save(out, name)
            else:
                self._save(output, name)
                            

    @contextlib.contextmanager
    def observe(self, model: torch.nn.Module):
        handles = {}
        for name, modu in model.named_modules():
            def hook(module, input, output, name=name):
                if hasattr(module, '__observe__'):
                    self._outputs[name] = module.__observe__(input, output)
                        
            handle = modu.register_forward_hook(hook)
            handles[name] = handle
        try:
            yield model
        finally:
            for _, handle in handles.items():
                handle.remove()
    

class GradObserver(BaseObserver):
    def __init__(self) -> None:
        self._grad_norms = OrderedDict()
        self._grad_vectors = OrderedDict()
        self._metric_tensors = defaultdict(lambda: defaultdict(lambda: torch.tensor(0.0)))
        self._norm_tensors = defaultdict(lambda: torch.tensor(0.0))
        self._n = 0

    def _reset(self):
        self._grad_norms.clear()
        self._grad_vectors.clear()

    def _save_grad(self, name):
        def __save_grad(grad: torch.Tensor):
            with torch.no_grad():
                self._grad_norms[name] = grad.detach().square().mean().sqrt()

                zero = None
                mean = None
                for i in range(grad.ndim):
                    _zero = grad.transpose(0, i).flatten(1).select(1, 0)
                    _mean = grad.transpose(0, i).flatten(1).mean(dim=1)
                    if zero is None:
                        zero = _zero
                        mean = _mean
                    else:
                        zero = torch.cat([zero, _zero], dim=0)
                        mean = torch.cat([mean, _mean], dim=0)

                self._grad_vectors[name] = {
                    'vec_zero': zero,
                    'vec_mean': mean,
                    'sca_zero': grad.flatten().select(0, 0),
                    'sca_mean': grad.mean(),
                }
            return grad
        return __save_grad

    def _calculate_metric_tensor(self):
        with torch.no_grad():
            zero = None
            mean = None
            for key, vecs in self._grad_vectors.items():
                self._metric_tensors[key]['zero'] = \
                    self._metric_tensors[key]['zero'] \
                        + vecs['vec_zero'].unsqueeze(0)*vecs['vec_zero'].unsqueeze(1)
                self._metric_tensors[key]['mean'] = \
                    self._metric_tensors[key]['mean'] \
                        + vecs['vec_mean'].unsqueeze(0)*vecs['vec_mean'].unsqueeze(1)
                
                if zero is None:
                    zero = vecs['sca_zero'].unsqueeze(0)
                else:
                    zero = torch.cat([zero, vecs['sca_zero'].unsqueeze(0)], dim=0)

                if mean is None:
                    mean = vecs['sca_mean'].unsqueeze(0)
                else:
                    mean = torch.cat([mean, vecs['sca_mean'].unsqueeze(0)], dim=0)

            self._metric_tensors['__all__']['zero'] = \
                self._metric_tensors['__all__']['zero'] \
                    + zero.unsqueeze(0)*zero.unsqueeze(1)
            self._metric_tensors['__all__']['mean'] = \
                self._metric_tensors['__all__']['mean'] \
                    + mean.unsqueeze(0)*mean.unsqueeze(1)
                
            for key, norm in self._grad_norms.items():
                self._norm_tensors[key] = self._norm_tensors[key] + norm

        self._reset()
        self._n += 1

    def save(self):
        fig, ax = plt.subplots(1, 1)
        norm_tensors = torch.tensor(list(self._norm_tensors.values()))
        ax.plot(torch.log2(norm_tensors / self._n))
        ax.set_title("Grad Norm")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Log2 Norm")
        ax.set_xticks(range(len(self._norm_tensors)))
        ax.set_xticklabels(list(self._norm_tensors.keys()), rotation=90)
        fig.tight_layout()
        wandb.log({f"Grad/LogNorm": wandb.Image(fig)})


        for name, x in self._metric_tensors.items():
            fig, ax = plt.subplots(2, 1)

            zero = x['zero']
            var = torch.diag(zero)
            zero = zero.add(1e-6) / torch.sqrt(var.unsqueeze(0)*var.unsqueeze(1)).add(1e-6)
            zero = zero.detach().cpu()

            ax[0].imshow(zero, vmin=-1, vmax=1)
            ax[0].set_title(f"{name} Zero")
            if name == '__all__':
                ax[0].set_yticks(range(len(self._norm_tensors)))
                ax[0].set_yticklabels(list(self._norm_tensors.keys()))
            else:
                ax[0].set_yticks([])
            ax[0].set_xticks([])

            mean = x['mean']
            var = torch.diag(mean)
            mean = mean.add(1e-6) / torch.sqrt(var.unsqueeze(0)*var.unsqueeze(1)).add(1e-6)
            mean = mean.detach().cpu()

            ax[1].imshow(mean, vmin=-1, vmax=1)
            ax[1].set_title(f"{name} Mean")
            if name == '__all__':
                ax[1].set_yticks(range(len(self._norm_tensors)))
                ax[1].set_yticklabels(list(self._norm_tensors.keys()))
            else:
                ax[1].set_yticks([])
            ax[1].set_xticks([])

            fig.tight_layout()
            wandb.log({f"Grad/{name}": wandb.Image(fig)})

    @contextlib.contextmanager
    def observe(self, model: torch.nn.Module):
        # 1. Register forward_hook fn to save the output from specific layers
        handles = {}

        for name, modu in model.named_modules():
            if isinstance(
                modu, 
                (
                    torch.nn.Linear,
                    torch.nn.Conv1d,
                    torch.nn.Conv2d,
                    torch.nn.Conv3d,
                    torch.nn.ConvTranspose1d,
                    torch.nn.ConvTranspose2d,
                    torch.nn.ConvTranspose3d,
                )
            ):
                handle = modu.weight.register_hook(self._save_grad(name))
                handles[name] = handle
        try:
            yield model
        finally:
            self._calculate_metric_tensor()
            for _, handle in handles.items():
                handle.remove()