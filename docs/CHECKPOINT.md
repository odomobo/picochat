Need to get a mixed dataset with fineweb-edu and 3rd grader educational material. This will give it examples of things it can know and things it can not know.

Also make a mix of fineweb-edu and tinystories, for training tiny models.

Need to make web interface so I can see logit distribution and conviction.

Need to create some n-shot example completions of things the model should be able to complete and should not.

Need to try to make sense of conviction. Probably need to try an architecture where there's another norm right before lm_head. Then cosine similarity will be a better representation of what's going on. I wonder if I need to do something to tame output_projection in that case. I'm not really sure if I need output projection at all as long as the network is deep enough... should play around with this some more.