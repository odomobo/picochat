I got tied weights working well with an output projection layer (hidden_dim x hidden_dim) that sits right before lm_head, and with reducing wte training rate to somewhere between 0.02 and 0.004 . However, training is basically identical to when the weights are untied, both in terms of training time and model performance.

However, this is actually beneficial for the next test I want to do! I want to try to implement next-token confidence with this architecture:

tied weights, with an output projection layer just before lm_head. But! there is also another output in addition to the logits - confidence. Confidence will be a linear projection (hidden_dim x 1), which will take the activations prior to the projection layer. I might need to do something with scaling... or I might not... there's an rms norm just before this hidden layer.

Anyhow, the idea is that the hidden layer just before the output projection layer will learn the next token prediction (not yet transformed into lm_head space) and also a confidence level. Then the linear projection extracts the confidence level, and the output proj extracts the lm_head next token prediction.

Ok, let's label the hidden activations. The hidden activation between rms norm and output proj is "h_pre". The hidden activation between output proj and lm_head is "h_post". The expected token is "t_exp". The output of the confidence projection is "o_conf".

How do we teach the confidence level? Well, after the forward pass, we capture "h_post". We're going to use this as the actual idea the model was trying to say. Then, we project "t_exp" backwards through lm_head (or simply project it forward through wte - same thing), in a 1-hot fashion. We capture this hidden state as "h_exp". So, we wanted the model to have the idea of "h_exp", but it actually had the idea of "h_post". We compare the cosine similarity between the two to determine the model's confidence! Then we backprop this expected confidence through "o_conf" backwards, which will feed through the confidence projection, and back through the network.

The important thing about this is that the tied wte/lm_head is able to cluster similar tokens together, and separate dissimilar tokens. There is reason to think this will happen, because similar concepts do seem to result in similar embedding spaces in practice, but if for whatever reason this is not the case, then the confidence level will be fairly useless.

Similarly, if the clustering is non-regular, then confidence will be weighted unevenly across different parts of conceptual space, thus reducing its utility.

Either way, I want to explore it, see how well it works in practice.


I've decided I need a different pretraining corpus for my tiny models. I've decided that somewhere between 50% and 90% of the pretraining corpus should be text which is relevant for a 3rd grader (9-year-old). The concepts they should know about the world, things they learn in school, and tests they would take in school. The rest can be a mixture of information relevant to all ages. This will give the model the ability to confidently learn the simple concepts, and be confident that it doesn't understand the more complicated things. It should learn how to parrot those things, but it'll also be aware that its parroting is of low confidence.

I'm not sure how to curate this? Maybe see if BabyLM has any resources on this. Do some searches...


I'm playing around with a couple different models:

a 10M tied model like:
4 layers
dim 256
~15 minutes (with ~10 minutes training time)

a 15M tied model:
4 layers
dim 384
~25 minutes (with ~20 minutes training time)

a 25M tied model:
4 layers
dim 512
~70 minutes (with ~60 minutes training time)

The 10M model is good for quick sanity checks.

I also want to work on a 100M model, which should take about 16 hours to train. I need to come up with a good architecture for it. This can be used for more thorough tests.

Then, I want to clean up and sanitize my stuff. This should be quick! Clean up my folders, but also I want to clean up my wandb logging.

Next, I want to use a different activation function instead of ReLU^2. I want to use GELU = x · sigmoid(1.702x). It's basically ReLU with a soft knee. And of course, swappable activation functions.

After that, I want configurable attention head size.

Then I want to play around with those.

Then, I think I need to create a "findings" document, which stays clean.

Instead of "queueing" runs, what I'll do is init a bunch of different runs which I want to queue, then I'll write a one-off bash script which will set/train/validate on each one. Then I fire and forget. Maybe each one sets the terminal window name or something, so I can see which step it's on.


I was interested in playing around with model growing, but it has a speedup of maybe 5x, which is kinda piddly. It just means there's a linear speedup for some things, but that's no good when the underlying architecture has problems. So instead of this, let's focus on confidence!!!!
