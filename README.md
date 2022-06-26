# HAVOK & Reservoir computing for chaotic dynamics forecast

<p align="center"><b>Group 2202</b> // Barone, Nagaro, Ninni, Valentini</p>

**Referee**: Prof. Jeff Byers, Naval Research Laboratory, Washington [linkedin](https://www.linkedin.com/in/jeff-byers-8458969/)

This is our final project for Laboratory of Computational Physics (module B). In this work, we review two dynamical systems analysis techniques and explore whether it is possible to use them in **chaotic dynamics forecast**.

The first technique is the **Hankel Alternative View Of Koopman** (HAVOK) analysis, based on a [paper](https://www.nature.com/articles/s41467-017-00030-8) by S. Brunton et al, 2017. At first, we develop the framework to achieve the same results shown in the paper; then we discuss some features of the new coordinate space.

The second technique in our review is **reservoir computing**. (...)

<br>

In this review, we probe the two techniques above to characterize the dynamics reconstruction accuracy & prediction capabilities. Our results are benchmarked on a Lorenz attractor system. Eventually, we provide a demo which uses HAVOK to issue a trigger that prevents the Lorenz attractor to switch lobes. To achieve this, we train a **Reinforcement Learning** model to interact with the Lorenz system.

### HAVOK sentinel reinforced model

The following demo implements a trigger for chaotic dynamics control. The acting model is a Deep Deterministic Policy Gradient built in [Keras](https://keras.io/examples/rl/ddpg_pendulum/), whereas the sentinel model is a thresholded HAVOK coordinate. In other words, when the coordinate computed through the HAVOK analysis (on a moving window) exceeds a given threshold, the actor model is triggered to execute an action that prevents the Lorenz attractor to switch lobes.

<iframe width="560" height="315" src="https://www.youtube.com/embed/KdFz_q_qo3w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br>

### usage of rhavok

**rhavok** is a small library we use to collect all the common routines required by this work. You can install the library in *development mode* running the following command from current directory:
```bash
pip install -e ./lib/
```
Then, the library will by available on your system through the usual *import* fashion:
```python
import rhavok
```
You may find some documentation on it, provided as Jupyter notebooks (`doc_*.ipynb`) inside the `lib_rhavok` folder.

<br>

## The HAVOK analysis workflow

![workflow_havok](./img/workflow_havok.svg)

## The Reservoir Computing workflow

(...) insert picture here

***

<h5 align="center">Lab of Computational Physics (module B) project<br>AY 2021/2022 University of Padua</h5>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62724611/166108149-7629a341-bbca-4a3e-8195-67f469a0cc08.png" alt="" height="70"/>
  &emsp;
  <img src="https://user-images.githubusercontent.com/62724611/166108076-98afe0b7-802c-4970-a2d5-bbb997da759c.png" alt="" height="70"/>
</p>
