# to fix

## things to try later:
- vae (variational autoencoder) instead of raw pixels + sac

versions

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
    - 