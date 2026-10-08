# to fix

## things to try later:
- vae (variational autoencoder) instead of raw pixels + sac
    - train on real + sim
    - remove noise (edges, top bottom) to keep center important information
    - randomize lighting, colors, textures, camera noise, blur, etc
    - goal is to merge real images and sim images into one latent spaces to seamless sim-to-real
- increase max_cte to allow car to stray a bit further from track to optimize racing lines

versions

> note: changes only encompass the major changes, small extra changes may also have been made


## v4:

### changes
1. add a second car as an opponent
    - use a basic pretrained model
2. optimize for being ahead and overtakes
    - huge reward for overtaking, survival bonux for being in 1st
    - no information on which car is ahead, so we rely on lap time

- **results**
    -


## v3:

### changes:
1. lap times (only change for now, add more based on initial results)
    - change reward to minimize lap times instead of just keeping center line
    - previous rewards and losses (straying from center, speed, etc) will remain, but be dampened
    - majority is for improving lap times

- **results**
    - 9.92 second average lap time
    - now testing:
        - testing generalization on different tracks
            - does not generalize, expected because we are currently only using raw frames
        - non-probalistic version to see if it is smooth
            - non stochastic version doesnt work nearly as well, model learned to work with noise
        - adding an opponent (v4)


## v2:

### changes:
1. speed in reward
    - scale the centering reward by speed
    - going fast and in center gives high rewards than slow and in center
    - small penalty for extremely slow speed

2. replace raw returns with actor critic
    - use an actor critic to evaluate how good this action was
    - PPO for now, SAC later 

3. fix how returns are computed at episode boundaries
    - when computing discontinued returns, future is reset on terminated and not truncated
    - data from the next episode can leak backwards
    - bootstrapping and terminating vs truncated

- **results**: 
    - ran it overnight for run 1
    - logs showed very good results, laps finished, but ran out of memory due to extended episodes
    - when i tried in the testing script, it barely got past the first turn
    - will train a second time

    - after some fixes, i resumed training and it actually completed laps pretty consistently
    - going to add lap finishes as large rewards

    - extremely consistent lap completions, optimize for lap time next


## v1:
- very basic
- only images as input
- **results**:
    - very poor performance, model stops learning and barely moves to maximize center reward