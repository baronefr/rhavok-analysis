# **HAVOK analysis targets**

- **scalability**: how does the Havok scale with different parameters?
    - change number of 'training' samples (time samples used to build Hankel)
    - change Hankel window size (q)
- **threshold tuning**
    - one forcing term threshold
        - try more ranks
        - what about multiple ranks combined to one?
    - two forcing term threshold
- **CNN** on Havok
    + stats on moving window
    + balance datasets
- other things?

### rhavok library

- add dump/reload functionality
- make moving window test the most general as possible