# Findings

When tying wte to lm_head, training was degraded significantly until I added a projection matrix (without residuals).

When conviction doesn't directly train lm_head, then conviction loss seems steady. I have suspicion that this is related to irregularly-distributed embedding space (so losses in one part of embedding space will have a different scale from losses in another part). I want to try letting the conviction head train the lm_head, and see if it helps to regularize the distribution.

# Research

## Why I used a projection matrix without residuals when weight tying
Using the Output Embedding to Improve Language Models
Ofir Press, Lior Wolf
https://arxiv.org/abs/1608.05859


## To analyze: why weight tying should conceptually be reasonable
Why and when tying embedding (a story)
https://www.reddit.com/r/MachineLearning/comments/1eqm0lr/r_why_and_when_tying_embedding_a_story/
Note: there's also an associated paper, so I should probably reference it once I make sense of it. https://openreview.net/forum?id=yyYMAprcAR

