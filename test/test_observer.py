# Test torchobserver/observer.py by running:
# pytest test/test_observer.py

import pytest
import torch, torchaudio
from torchobserver import Observer, GradObserver
from matplotlib import pyplot as plt
import wandb

def test_observer_audio():
    wandb.init(project='test')
    observer = Observer(sr=16000)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return x
        
        def __observe__(self, input, output):
            x = output
            return {'type': 'Audio', 'name': 'audio', 'data': x, 'kwargs': {}}
        
    model = Model()

    with observer.observe(model):
        model(torch.randn(16000))
    observer.save()

def test_observer_image():
    wandb.init(project='test')
    observer = Observer()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return x
        
        def __observe__(self, input, output):
            x = output
            return {'type': 'Image', 'name': 'image', 'data': x, 'kwargs': {}}
        
    model = Model()

    with observer.observe(model):
        model(torch.randn(64, 64, 3))
    observer.save()

def test_observer_mel():
    wandb.init(project='test')
    def vocoder(x):
        return torchaudio.transforms.GriffinLim(n_fft=1024)(x)
    observer = Observer(vocoder=vocoder, sr=16000)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return x
        
        def __observe__(self, input, output):
            x = output
            return {'type': 'Mel', 'name': 'mel', 'data': x, 'kwargs': {}}
        
    model = Model()

    with observer.observe(model):
        model(torch.randn(513, 100))
    observer.save()

def test_observer_figure():
    wandb.init(project='test')
    observer = Observer()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return x
        
        def __observe__(self, input, output):
            x = output

            fig, ax = plt.subplots(1, 1)
            ax.plot(x)

            return {'type': 'Image', 'name': 'figure', 'data': fig, 'kwargs': {}}
        
    model = Model()

    with observer.observe(model):
        model(torch.randn(100))
    observer.save()

def test_gradobserver_linear():
    wandb.init(project='test')
    observer = GradObserver()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
            )
            
        def forward(self, x):
            return self.linear(x)
        
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for _ in range(10):
        with observer.observe(model):
            x = torch.randn(1, 10)
            y = model(x)
            loss = loss_fn(y, torch.randn(1, 10))
            loss.backward()
            optimizer.zero_grad()

    observer.save()

def test_gradobserver_conv():
    wandb.init(project='test')
    observer = GradObserver()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3),
            )
            
        def forward(self, x):
            return self.conv(x)
        
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for _ in range(10):
        with observer.observe(model):
            x = torch.randn(1, 3, 64, 64)
            y = model(x)
            loss = loss_fn(y, torch.randn(1, 32, 58, 58))
            loss.backward()
            optimizer.zero_grad()

    observer.save()