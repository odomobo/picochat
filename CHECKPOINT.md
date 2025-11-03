I tied weights together, and just using tied weights (with no more transformer layers) significantly hurts performance. I guess the lm_head was doing some heavy lifting. I want to see if I can get at least equiv performance out of a network with a total size of 15M, because otherwise I'm worried something is wrong...

I'm trying a 15M model like:
tied weights - yes
dim 384
4 layers
everything else default
playing around with learning rate

Maybe play around with learning rates. I used tied_weights_lr of 0.2, but it should be somewhere between 0.2 and 0.004, so a lot of room for exploration.

Next, I want to use a different activation function instead of ReLU^2. I want to use GELU = x · sigmoid(1.702x). It's basically ReLU with a soft knee. And of course, swappable activation functions.

After that, I want configurable attention head size.