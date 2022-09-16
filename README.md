# CaMeLS: Cooperative Meta-Learning Service for Recommender Systems

---

# Instructions

### Installation
1. Clone the repository and its submodules with `git clone --recurse-submodules`.
2. Recommended: Build the environment docker image with `docker compose build`. 
Alternatively: Install the requirements to a new or existing Python 3.9 virtual environment.
3. Skip this step if you want to use this release version of CaMeLS instead of configuring your own.
Instructions to host a new CaMeLS server are in [the server readme](camels/server/README.md).
Instructions to connect with the new CaMeLS server are in [the client readme](camels/client/README.md).
4. Write the CaMeLS server address `TBA` to the file `camels/client/connection_settings.py`.
5. Optional: Use the RecSys data loader to load and process common data sets.
Refer to [its readme](camels/data_loader/README.md) for usage.
6. Use `camels/client_functions.py` to connect with the server and execute CaMeLS routines.
Or call functions in `camels/client/client_routine.py` directly to customize more options, e.g., use custom data sets.

---

### You can find the CaMeLS prototype that is discussed in our paper in the workshop_perspectives branch. The instructions above do not apply to that version.
