This Color Neural Field code was written by Anna SONG (last modified: Fri Dec 14 2018),
and corresponds to our article:

Song A, Faugeras O, Veltz R (2019)
A neural field model for color perception unifying assimilation and contrast.
PLoS Comput Biol 15(6):e1007050.
https://doi.org/10.1371/journal.pcbi.1007050

If you use it, please cite us. The code is described in S2 Appendix of the article.

It contains:
- settings.py, which determines all the important settings for the color neural field (CNF) and its regression,
in particular the dataset to which the model is regressed. It also contains synthetic datasets similar to that of Monnier & Shevell (2004 and 2008). 
- HSL_data.py contains our data for the color shifts measured inside the HSL chromatic disk
- initialization.py, containing various parameter values for q, which can serve as starting values;
- visualize.py, with the direct simulation of the CNF dynamics, and various displaying functionalities
- main_pytorch.py, which performs the regression to the datasets of Monnier & Shevell (2004 and 2008)
- main_HSL_pytorch_2D.py, which performs the regression with a 2D color space to our dataset (referred as 2017 here)
- pytorch_argmax.py defines the functions for the soft argmax and max used insde the two previous scripts.


If you want to help improve the code, please write to anna.song.maths@gmail.com

Thank you for downloading!
