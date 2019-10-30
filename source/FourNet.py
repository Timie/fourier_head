'''
This file contains exploration of possibilities of 'Fourier Net' - my
humble attempt to 'invent' neural network architecture that is robust
to overfitting. The idea is based on human laziness - a cascade of
models about observed reality form simplest (easiest to think about)
to most complex (very hard to think about), and weighting of the 
said models (simple model = high weight, complex model = low weight)
similar to weighting of 'rotating vectors' in spectral decomposition
of a signal (calculated by Fourier transform).

The effect this should lead to is that simple heads should give 
straight answer about simple problems, and give undecisive answers
about more complex solutions - otherwise their incorrect guess
would overweight the correct guesses of more complex networks.

More info in readme...

The file is protected by MIT licence, which should reside somewhere
in the root folder of the repo. If it's not there, just give me 
a shout, and I will provide it.

MIT Licence
(c) 2019 Adam Babinec (Timie)

'''

from fastai.vision import *
from fastai.vision.learner import num_features_model,create_body
import torch

def simple_cnn_max(actns:Collection[int], kernel_szs:Collection[int]=None,
               strides:Collection[int]=None) -> nn.Sequential:
    """CNN with `conv2d_relu` layers defined by `actns`, `kernel_szs` and `strides`
    This function is kindly stolen from fast.ai doc: https://docs.fast.ai/layers.html
    """
    nl = len(actns)-1
    kernel_szs = ifnone(kernel_szs, [3]*nl)
    strides    = ifnone(strides   , [2]*nl)
    layers = [conv_layer(actns[i], actns[i+1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(nn.Sequential(nn.AdaptiveMaxPool2d(1), Flatten()))
    return nn.Sequential(*layers)

class FourierHead(torch.nn.Module):
    '''
    Represents a head of a simple neural network, which has multiple branches.
    Branches have different size of bottleneck, and their results are weight-summed
    to give the final results. Narrower the bottleneck, higher the weight. This
    is to favour simple models instead of more complex ones. As an effect:
    - simple model are likely to be used for result
    - if simle model cannot explain the result, it should give undecisive result,
      and the final decision is on the more complex model.

    There are multiple models and each is twice as big as the previous one,
    up to the 'H' width. The weight difference between thinner and wider
    model is defined by weightDecay.

    Disclaimer: I have no idea whether this architecture of the branches is
    actually effective - maybe some other combination of layers would give
    better performance. The intended feature - a bottleneck of different
    sizes - is there though, and that's what we want to observe.

    In the future, there could be another parallel "recaller" set of heads that would
    try to to remember the input. If they recall the input quite well, they would
    set the weight of the corresponding models higher. If they do not recall the input,
    only the simplest (?) model will be used to generate the prediction.
    '''
    def __init__(self, D_in, H, D_out, levels, weightDecay=2):
        super(FourierHead, self).__init__()
        self.branches = []
        self.weights = []
        self.total_weight = 0
        for level in range(levels):
            weight = weightDecay**(-level)
            self.total_weight += weight
            bottleneck = int(H / (2**(levels-level-1)))
            branch = torch.nn.Sequential(
                                 Flatten(),
                                 torch.nn.Dropout(0.2),
                                 torch.nn.Linear(D_in, bottleneck),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(bottleneck, D_out),
                                 torch.nn.LogSoftmax(dim=1))
            self.branches.append(branch)
            self.weights.append(weight)
    
    def forward(self, x):
        y = None
        for branch, weight in zip(self.branches, self.weights):
            if y is None:
                y = branch(x) * weight
            else:
                y += branch(x) * weight
        return y / self.total_weight

def fourier_cnn(actns:Collection[int], headInputSize=32, headBottleNeck=64, headOutputSize=10, headLevels=5, weightDecay=2) -> nn.Sequential:
    'CNN net with fourier head'
    nl = len(actns)-1
    kernel_szs = [3]*nl
    strides    = [2]*nl
    layers = [conv_layer(actns[i], actns[i+1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(nn.Sequential(FourierHead(headInputSize, headBottleNeck, headOutputSize, headLevels, weightDecay), Flatten()))
    return nn.Sequential(*layers)



if __name__ == '__main__':

    # I cannot make CUDA work on my laptop + python. Therefore,
    # I have to test it on very simple data - MNIST datasets.
    mnistData = untar_data(URLs.MNIST)
    tfms = get_transforms(do_flip=False)

    data = (ImageList.from_folder(mnistData)
        .split_by_rand_pct(valid_pct=0.3, seed=42)          
        .label_from_folder()
        .transform(tfms)
        .databunch())

    epochs = 20

    # Try resnet model with Fourier head. Just out of curiosity.
    resnetModel = models.resnet18
    body = create_body(resnetModel, True)
    nf = num_features_model(nn.Sequential(*body.children()))
    head = nn.Sequential(FourierHead(nf, 64, 10, 5, math.sqrt(2)))
    resLearner = cnn_learner(data, resnetModel, pretrained=True, metrics=[accuracy], 
                             custom_head=head)
    resLearner.fit(int(epochs/5)) # Less epochs, as this is too slow...
    resLearner.save('fourier_resnet')

    # Very simple model model with weightDecay = 2
    model2 = fourier_cnn((3,16,16,2), 32, 64, 10, 5)
    learner = Learner(data, model2, metrics=[accuracy])
    learner.fit(epochs)
    learner.save('fourier_5levels')

    # Try the same model with weightDecay = sqrt(2). That should
    # solve issue when simples model actually dominates the results 
    # completelly, and gives all other models power to overpower
    # the simplest model.
    model3 = fourier_cnn((3,16,16,2), 32, 64, 10, 5, math.sqrt(2))
    learner2 = Learner(data, model3, metrics=[accuracy])
    learner2.fit(epochs)
    learner2.save('fourier_5levels_lowdecay')
    learner2.validate()

    # Simpler baseline - as wide, as the most complex model in
    # Fourier head. Note, we use only one-level fourier head
    # (= with one branch)
    modelBaseline = fourier_cnn((3,16,16,2), 32, 64, 10, 1)
    learnerBaseline = Learner(data, modelBaseline, metrics=[accuracy])
    learnerBaseline.fit(epochs)
    learnerBaseline.save('fourier_baseline')

    # More complex baseline - as wide as all models in the above
    # fourier models. Note, we use only one-level fourier head
    # (= with one branch)
    modelBaseline2 = fourier_cnn((3,16,16,2), 32, 64+32+16+8+4, 10, 1)
    learnerBaseline2 = Learner(data, modelBaseline2, metrics=[accuracy])
    learnerBaseline2.fit(epochs)
    learnerBaseline2.save('fourier_baseline2')
    learnerBaseline2.validate()
