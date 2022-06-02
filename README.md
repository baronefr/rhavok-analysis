# reservoir/HAVOK analysis

This is our final project for Laboratory of Computational Physics (module B). In this work, we review some latest dynamical systems analysis techniques.

The first technique is the *Hankel Alternative View of Koopman* (HAVOK) analysis, based on a [paper](https://www.nature.com/articles/s41467-017-00030-8) by S. Brunton et al, 2017. We implement the same results shown in the paper and discuss some features of the forcing term $v_r$.

The second technique in our review is **reservoir learning**. (...)

In the review, we probe the two techniques above, to characterize the dynamics reconstruction accuracy & prediction capabilities. Our results are focused on a Lorenz attractor system. Eventually, we provide a real time demo which uses HAVOK & reservoir learning to issue a runtime trigger that prevents the Lorenz attractor to switch lobes. Just for the example purpose, we use a reinforcement learning model to interact with the Lorenz system.


**Students** working on this project:
| Group 2202  |
| ------------- |
| Barone Francesco Pio |
| Valentini Lorenzo |
| Nagaro Gianmarco | 
| Ninni Daniele |

**Referee**: Prof. Jeff Byers, Naval Research Laboratory, Washington [linkedin](https://www.linkedin.com/in/jeff-byers-8458969/)

### usage of rhavok

**rhavok** is a small library we use to collect all the common routines required in this work. You can install the library in *development mode* running the following command from current directory:
```
pip install -e ./lib_rhavok/
```
Then, the library will by available on your system through the usual import fashion:
```
import rhavok
```

## The HAVOK analysis workflow

![workflow_image](./img/workflow.svg)

## The reservoir learning workflow

(...)

## reservoir and HAVOK as triggers for chaotic dynamic control

(...)

***

<h5 align="center">Lab of Computational Physics (module B) project<br>AY 2021/2022 University of Padua</h5>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62724611/166108149-7629a341-bbca-4a3e-8195-67f469a0cc08.png" alt="" height="70"/>
  &emsp;
  <img src="https://user-images.githubusercontent.com/62724611/166108076-98afe0b7-802c-4970-a2d5-bbb997da759c.png" alt="" height="70"/>
</p>
