# Local Target 

- Apply Improved Mean Flow on Videos using wanvae as VAE

- Train on current 2k original videos from wansyn (available in `./cache`)

- Training Steps: 4,000

# Server Target

- Model size must be less than 300M

- Train on entire wansyn (from wan2.1) dataset

# General Requirements

- Put all hyperparams into `./config.py` and check `./config.py` to check what parameters can be tuned. Must must strictly follow my orders in `./config.py`.

- Changing `config.py` file to tune parameters

- Local training is for tuning parameters and scale to server training

- Loss curve should be firstly stable and fast-converging

- In no way can you change input size!!!

- Check training progress every 15 mins. Stop the node experiment once you find it clearly unstable or worse than current best. Don't output your answers if it goes well.
