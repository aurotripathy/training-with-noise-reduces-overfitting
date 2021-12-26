# Training With Noise to Reduces Overfitting

This example illustrates the smoothing effects of training with input jitter.

## The Dataset

The dataset has only 31 points distributed equally adross two classes. I came across it in Russell Reed's seminal book, Neural Smithing (page 282). The data isn't (digitally) available anywhere, so I had to recreate it by hand (it was fun to work once again with a ruler and pencil). The two classes are resresented by the '+' and 'o' symbols. 

<img src="hand-derived-point.png-1.png" alt="drawing" style="width:600px;"/>

Going through an analog to digital conversion, we get a dataset of 31 point across two classes:

<img src="original-dataset.png" alt="drawing" style="width:700px;"/>

## The Model

The model is a very simple 2/50/10/1 MLP network, the same used the Russell's book.

```
class ThreeLayerNLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 10)
        self.layer2 = torch.nn.Linear(10, 50)
        self.output = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        out = torch.sigmoid(self.output(x))
        return out
```

## Trained to Intentionally Overfit

Per Russell, "With 671 weights, but only 31 training points, the network is very underconstrained and chooses a very nonlinear boundary". And it does turn out that way as you can see below.

<img src="Known-Overfit.png" alt="drawing" style="width:600px;"/>

## Smoothing with Jitter

Per the book, training with jitterred data discourages sharp changes in the response near the training points and so discurages the network from overly complex boundaries. 

<img src="Noise-Added-to-Smooth-boundary.png" alt="drawing" style="width:600px;"/>
