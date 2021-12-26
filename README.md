# Training With Noise to Reduces Overfitting

This example illustrates the smoothing effects of training with input jitter.

## The Dataset

The dataset has only 31 points and two classes. I came across it in Russell Reed's seminal book, Neural Smithing (page 282). The data isn't (digitally) available anywhere, so I had to recreate it by hand (it was fun to work once again with a ruler and pencil). The two classes are resresented by the '+' and 'o' symbols. 

<img src="hand-derived-point.png-1.png" alt="drawing" style="width:600px;"/>

Going through an analog to digital conversion, we geta dataset of 31 point and two classes:

<img src="original-dataset.png" alt="drawing" style="width:600px;"/>

## The Model

The model is a very simple 2/50/10/1 network, the same used the Russell's book. 

<code>
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
</code>


## Trained to Intentionally Overfit

## Smoothing with Jitter
