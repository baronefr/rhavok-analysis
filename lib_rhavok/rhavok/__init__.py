import rhavok.gym
import rhavok.havok
import rhavok.systems
import rhavok.utils

###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python library implements the HAVOK analysis & reservoir learning
#  routines used for our University project.
#
#  coder: F. Barone (manteiner) + G. Nagaro, D. Ninni, L. Valentini
#--------------------------------------------------------------------------
#  Open Access licence
#--------------------------------------------------------------------------

__version__ = '0.1.4'
__major_review__ = '01 june 2022'

def version():
    print('rhavok | v', __version__)
    print(' major review:', __major_review__)

def credits():
    print('rhavok | v', __version__)
    print(' Barone, Nagaro, Ninni, Valentini')
    print(' www.github.com/baronefr/rhavok-analysis')