
# Fourier Head
This repo contains a very basic attempt to create a neural network head that would be able to avoid overfitting without any 'supervision' (= no need for validation dataset).

## Motivation
The idea behind the design is based on realisation that people (or in general, animals) are very good at generalisation because they need to conserve energy, and therefore they look for the concepts (models) that are least power-hungry = easy to think about = simple.

Therefore, the network should favour simple explanations of the observed data, and if there is no explanation it should go for some other "exceptional" explanation - one step more complex, and if that is not enough, it should pay more effort for another level of complexity.

But how can we make the network favour simple models? Just weight them! Simple concepts would have greater weight and more complex models would have less weight. The weight would decay with increasing complexity.

This brings a parallel to a nice 'rotating vectors' of spectral decomposition of a signal using Fourier transformation (for better understanding, see this nice video https://www.youtube.com/watch?v=r6sGWTCMz2k by 3blue1brown), where "rough" "big" slowly rotating vectors (simple models of observed data) have the highest amplitude, and quickly rotating vectors (describing tiny details) have small amplitude.

## Architecture
Fourier Head consists of N branches of the same sequential architecture. They share the same number of inputs and outputs, and contain one internal layer representing a 'bottleneck'. This bottleneck layer has different width in each branch. As the width of the bottleneck increases, its weight decreases (by a coefficient called 'weight decay'). The results of the branches (after log-softmax) are combined together by a wighted-sum to provide final result.

For more info about implementation, see the (very short and simple) source.

## Initial results
The initial commit contains 5 networks. 1 resnet based network with *Fourier Head* (just for curiosity), 2 generic CNN networks with *Fourier Head* (with 5 levels) with different weighting change between 'levels' and 2 regular networks with the same architecture and simple head. You can see that among the generic CNN nets, the one with *Fourier Head* and weight decay of sqrt(2) is performing the best.
### Generic CNN with 5-level Fourier Head and wieght decay = 2
      epoch     train_loss  valid_loss  accuracy  time
    0         1.925441    1.796517    0.521286  02:17
    1         1.726003    1.571653    0.544905  02:08
    2         1.594631    1.412261    0.586476  02:08
    3         1.504131    1.320040    0.607095  02:45
    4         1.457456    1.234528    0.618952  02:51
    5         1.373197    1.160555    0.638048  02:29
    6         1.347420    1.130665    0.652143  02:37
    7         1.330218    1.090021    0.659762  02:09
    8         1.297674    1.057122    0.675571  02:08
    9         1.278044    1.046029    0.675762  02:08
    10        1.235438    1.077177    0.670619  02:08
    11        1.254140    1.018779    0.687048  02:08
    12        1.256022    1.038485    0.674476  02:49
    13        1.228795    1.021069    0.684286  02:50
    14        1.243961    1.026611    0.680286  02:09
    15        1.225775    1.026422    0.680238  02:07
    16        1.208959    0.998195    0.680714  02:08
    17        1.203411    0.969051    0.696905  02:07
    18        1.216561    0.975645    0.696000  02:07
    19        1.201570    0.995539    0.687857  02:05
### Generic CNN with 5-level Fourier Head and wieght decay = sqrt(2)

    epoch     train_loss  valid_loss  accuracy  time
    0         1.869108    1.768944    0.523381  02:05
    1         1.618286    1.492403    0.601286  02:06
    2         1.471740    1.281370    0.669524  02:04
    3         1.362881    1.181301    0.682952  02:06
    4         1.273377    1.076135    0.691810  02:05
    5         1.238464    1.054197    0.696571  02:08
    6         1.188932    0.985831    0.710952  02:06
    7         1.156316    0.982355    0.705143  02:06
    8         1.143880    0.949778    0.714190  02:07
    9         1.126080    0.936264    0.710857  02:05
    10        1.122097    0.922585    0.716667  02:05
    11        1.115745    0.895013    0.725095  02:05
    12        1.113624    0.891038    0.730238  02:04
    13        1.071405    0.876110    0.734381  02:05
    14        1.072720    0.874370    0.731381  02:05
    15        1.064348    0.882119    0.730667  02:06
    16        1.056260    0.887527    0.725619  02:06
    17        1.076726    0.857975    0.739381  02:06
    18        1.089743    0.876004    0.727476  02:05
    19        1.088866    0.869615    0.732000  02:05

### Generic CNN with normal head of same width as the most complex fourier branch

    epoch     train_loss  valid_loss  accuracy  time
    0         1.777474    1.576030    0.483333  02:03
    1         1.587554    1.370753    0.547286  02:04
    2         1.491522    1.279103    0.575000  02:04
    3         1.424199    1.215621    0.604286  02:03
    4         1.384275    1.189376    0.604952  02:04
    5         1.353912    1.137516    0.629524  02:03
    6         1.317726    1.091645    0.643191  02:04
    7         1.290571    1.066082    0.652190  02:05
    8         1.282036    1.064583    0.648048  02:03
    9         1.271545    1.030724    0.664905  02:03
    10        1.239017    1.030304    0.665762  02:04
    11        1.227937    1.046896    0.657095  02:03
    12        1.245863    1.017997    0.670000  02:04
    13        1.234678    1.015286    0.668905  02:04
    14        1.209797    1.036319    0.657524  02:03
    15        1.209407    0.995398    0.676619  02:04
    16        1.213357    0.994032    0.684143  02:03
    17        1.209846    1.010906    0.672286  02:05
    18        1.210837    1.005037    0.677429  02:03
    19        1.200136    0.991307    0.684190  02:03

### Generic CNN with normal head of same width as all fourier branches together

    epoch     train_loss  valid_loss  accuracy  time
    0         1.688783    1.575152    0.502667  02:03
    1         1.445742    1.299377    0.580143  02:03
    2         1.340073    1.154797    0.622238  02:04
    3         1.278926    1.147242    0.619429  02:04
    4         1.259218    1.085633    0.648190  02:04
    5         1.268042    1.044267    0.662810  02:04
    6         1.238510    1.048727    0.655762  02:03
    7         1.234198    1.028184    0.663762  02:04
    8         1.185117    1.009769    0.672810  02:03
    9         1.224397    1.000055    0.673381  02:04
    10        1.187547    0.994930    0.683000  02:03
    11        1.192982    0.996813    0.679238  02:04
    12        1.199516    1.009796    0.669000  02:03
    13        1.161931    0.977072    0.678810  02:03
    14        1.187021    0.996873    0.677952  02:03
    15        1.162808    0.965597    0.683286  02:04
    16        1.193745    0.961769    0.688190  02:04
    17        1.176655    1.002474    0.670952  02:04
    18        1.168087    0.993894    0.675381  02:04
    19        1.159108    0.967017    0.684762  02:04
    
    
### Resnet18-based network with Fourier Head (just to get an idea)
    epoch     train_loss  valid_loss  accuracy  time
    0         1.900392    1.829581    0.604476  7:34:31
    1         1.396325    1.296008    0.827857  12:20
    2         1.100570    1.014531    0.876905  11:58
    3         0.908900    0.809344    0.935952  12:11

