# rhavok

**reference version**: 0.1.4
<br>**last update**: 2 Jun 2022

**rhavok** is a small library we use to collect all the common routines required by this work. You can **install** the library in *development mode* running the following command from current directory:
```
pip install -e .
```
Before installing, be sure that `setuptools` is updated to the latest version. Tested version `62.3.2`.

The library will by available on your system through the usual import fashion:
```
import rhavok
```
Because of development mode, all the changes committed to the rhavok subfolder will be effective the next time the library is imported by python interpreter.

#### Documentation

You may find some usage examples as Jupyter notebooks in this folder as `doc_*.ipynb`.



### Modules
```
-- gym         gym environments
-- havok       routines for havok analysis
-- systems     dynamical systems definitions
-- utils       various things
```

### Changelog

- **ver 0.1.4**: first implementation of rhavok as a library package.

***

Library manteined by *Francesco Barone*.
<br>The credit for the code goes to the entire **Workgroup**: Barone Francesco, Nagaro Gianmarco, Ninni Daniele, Valentini Lorenzo.
