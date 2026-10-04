# to fix

## things to try later:
- vae (variational autoencoder) instead of raw pixels + sac
- increase max_cte to allow car to stray a bit further from track to optimize racing lines

versions

> note: changes only encompass the major changes, small extra changes may also have been made

## v1:
- very basic
- only images as input
- **results**:
    - very poor performance, model stops learning and barely moves to maximize center reward

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