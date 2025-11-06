# Findings

When tying wte to lm_head, training was degraded significantly until I added a projection matrix (without residuals).

# Research

## Why I used a projection matrix without residuals when weight tying
Using the Output Embedding to Improve Language Models
Ofir Press, Lior Wolf
https://arxiv.org/abs/1608.05859


## To analyze: why weight tying should conceptually be reasonable
Why and when tying embedding (a story)
https://www.reddit.com/r/MachineLearning/comments/1eqm0lr/r_why_and_when_tying_embedding_a_story/
Note: there's also an associated paper, so I should probably reference it.

