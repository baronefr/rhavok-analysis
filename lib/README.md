# rhavok

**reference version**: 1.0.0
<br>**last update**: 25 June 2022

***

**rhavok** is a small library we use to collect all the common routines required by this work. You can **install** the library in *development mode* running the following command from current directory:
```bash
pip install -e .
```
Before installing, be sure that `setuptools` is updated to the latest version. Tested version `62.3.2`.

The library will by available on your system through the usual import fashion:
```python
import rhavok
```
Because of development mode, all the changes committed to the rhavok subfolder will be effective the next time the library is imported by python interpreter.


### Documentation

You may find some usage examples as Jupyter notebooks in this folder as `doc_*.ipynb`.

<br><br>

### Modules
```
-- gym         gym environments
-- havok       routines for havok analysis
-- systems     dynamical systems definitions
-- utils       various things
```

### Changelog

- **ver 0.1.4**: first implementation of rhavok as a library package
- **ver 1.0.0**: release

***

Library coded by *Francesco Barone*.
<br>The credit for the code goes to the entire **workgroup**: Barone Francesco, Nagaro Gianmarco, Ninni Daniele, Valentini Lorenzo.
