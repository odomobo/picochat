I got tied weights working well with an output projection layer (hidden_dim x hidden_dim) that sits right before lm_head, and with reducing wte training rate to somewhere between 0.02 and 0.004 . However, training is basically identical to when the weights are untied. This makes me wonder if there's any point to tying them, because it takes nearly the same compute either way (we have to perform forward passes and calculate gradients for both, either way).

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

Then, I think I want to go back to practicality. Since I want to be able to perform multiple runs, I need some kind of a way to queue runs, and another script that waits for new runs to be queued, and finally some way to manage which runs are queued. Some way to manage this anyhow.

Then, I need to create some script that keeps my computer awake for as long as it sees the GPU being taxed? So it'll check periodically, and if it's above maybe 10%, it'll keep it awake. Then I can queue runs, and the computer will stay awake until the runs are finished.

Then, I have the dream setup (I guess). At this point, I'll implement model growth. This will be tricky because I'll need a way to capture sub-runs within a run. I think there's already some functionality for "runs" within a base dir, so I think I can reuse and mutate that to work for me.

Once I've played around with growing (which will be a lot in itself, and probably pretty demoralizing because there's no way it's "good"):

Then I'll implement lora, and I'll play around with repeated lora, then merging, without growth. Then try lora with growth.

Then... I'm not sure what after this point