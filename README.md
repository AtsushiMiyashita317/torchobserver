# Setup
```bash
pip install git+https://github.com/AtsushiMiyashita317/torchobserver.git
```

# Usage
## Observe image
To observe an image using `torchobserver`, you can follow this example:

```python
import torch
from torchobserver import Observer

wandb.init(project='<your project name>')
observer = Observer()
class Model(torch.nn.Module):
    # === your model difinition === #

    def __observe__(self, input, output):
        """
        Args:
            input: Input of self.forward
            output: Output of self.forward
        Returns:
            dict or list of dict:
        """
        # === Procedure to get figure or tensor === #

        return {'type': 'Image', 'name': '<image name>', 'data': <figure or tensor>, 'kwargs': {}}
        # You can observe multiple images
        # return [
        #     {'type': 'Image', 'name': '<image name>', 'data': <figure or tensor>, 'kwargs': {}},
        #     {'type': 'Image', 'name': '<image name>', 'data': <figure or tensor>, 'kwargs': {}},
        # ]

    
model = Model()
# you can also observe submodules
# model = torch.nn.Sequential(
#     Model(),
#     Model()
# )

for i, batch in enumerate(dataloader):
    with observer.observe(model):
        model(batch)

    observer.save(key=i)

```


## Observe audio
To observe an audio using `torchobserver`, you can follow this example:

```python
import torch
from torchobserver import Observer

wandb.init(project='<your project name>')
observer = Observer(vocoder=vocoder, sr=sampling_rate)
class Model(torch.nn.Module):
    # === your model difinition === #

    def __observe__(self, input, output):
        """
        Args:
            input: Input of self.forward
            output: Output of self.forward
        Returns:
            dict or list of dict:
        """
        # === Procedure to get audio or acoustic feature === #

        return {'type': 'Image', 'name': '<image name>', 'data': <audio or acoustic feature>, 'kwargs': {}}
        # You can observe multiple images
        # return [
        #     {'type': 'Image', 'name': '<image name>', 'data': <audio or acoustic feature>, 'kwargs': {}},
        #     {'type': 'Image', 'name': '<image name>', 'data': <audio or acoustic feature>, 'kwargs': {}},
        # ]

    
model = Model()
# you can also observe submodules
# model = torch.nn.Sequential(
#     Model(),
#     Model()
# )

for i, batch in enumerate(dataloader):
    with observer.observe(model):
        model(batch)

    observer.save(key=i)

```

## Observe grad
To observe gradients using `torchobserver`, you can follow this example:

```python
    wandb.init(project='<your project name>')
    observer = GradObserver()   
        
    model = torch.nn.Sequential(
                torch.nn.Conv2d(...),
                ...,
                torch.nn.Conv1d(...),
                ...,
                torch.nn.Linear(...),
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for batch in dataloader:
        with observer.observe(model):
            x, y = batch
            y_ = model(x)
            loss = loss_fn(y_, y)
            loss.backward()
            optimizer.zero_grad()

    observer.save()
```
