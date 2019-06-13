#!/usr/bin/env python
# coding: utf-8
#
print(__doc__)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
import os
sns.set_style('darkgrid')
#
os.chdir('/Users/pauline/Documents/Python')
df = pd.read_csv("Tab-Morph.csv")
#
n = 25
x = df.profile # x is the same for both subplots 1 and 2 
rs = check_random_state(0)
y1 = df.sedim_thick + 80. * np.log1p(x) # y for subplot 1
y2 = df.Min + 3. * np.log1p(x) # y for subplot 2
#
# Fit IsotonicRegression and LinearRegression model for subplot 1
ir1 = IsotonicRegression()
y1_ = ir1.fit_transform(x, y1)
lr1 = LinearRegression()
lr1.fit(x[:, np.newaxis], y1)  # x needs to be 2d for LinearRegression
#
# Fit IsotonicRegression and LinearRegression model for subplot 2
ir2 = IsotonicRegression()
y2_ = ir2.fit_transform(x, y2)
lr2 = LinearRegression()
lr2.fit(x[:, np.newaxis], y2)  # x needs to be 2d for LinearRegression
#
# Parameters for 1 subplot
segments1 = [[[i, y1[i]], [i, y1_[i]]] for i in range(n)]
lc1 = LineCollection(segments1, zorder=0)
lc1.set_array(np.ones(len(y1)))
lc1.set_linewidths(np.full(n, 0.5))
# Parameters for 2 subplot
segments2 = [[[i, y2[i]], [i, y2_[i]]] for i in range(n)]
lc2 = LineCollection(segments2, zorder=0)
lc2.set_array(np.ones(len(y2)))
lc2.set_linewidths(np.full(n, 0.5))
#
# Plotting
fig = plt.figure(figsize=(8.8, 4.0), dpi=300) # figsize in inches
fig.suptitle('Isotonic regression plot, Mariana Trench, 25 profiles', 
             fontweight='bold', fontsize=12, x=0.5, y=0.99)
#
# subplot 1
ax = fig.add_subplot(121)
plt.plot(x, y1, 'r.', markersize=18, alpha=.5)
plt.plot(x, y1_, 'g.-', markersize=12, alpha=.5)
plt.plot(x, lr1.predict(x[:, np.newaxis]), 'b-', linewidth=.5)
plt.gca().add_collection(lc1)
plt.legend(('Geologic Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Geology: sediment thickness', fontsize=10, fontfamily='serif')
plt.xlabel('Bathymetric profiles, nr. 1-25', fontsize=10, fontfamily='sans-serif')
plt.ylabel('Sediment thickness, m', fontsize=10, fontfamily='sans-serif')
plt.annotate('A', xy=(0.1, .90), xycoords="axes fraction", fontsize=18,
           bbox=dict(boxstyle='round, pad=0.3', fc='w', edgecolor='grey', linewidth=1, alpha=0.9))
#
# subplot 2
ax = fig.add_subplot(122)
plt.plot(x, y2, '.', c = '#5654a2', markersize=18, alpha=.5)
plt.plot(x, y2_, '.-', c = '#b44c97', markersize=12, alpha=.5)
plt.plot(x, lr2.predict(x[:, np.newaxis]), 'b-', linewidth=.5)
plt.gca().add_collection(lc2)
plt.legend(('Bathymetric Data', 'Isotonic Fit', 'Linear Fit'), loc='lower left')
plt.title('Bathymetry: maximal depths', fontsize=10, fontfamily='serif')
plt.xlabel('Bathymetric profiles, nr. 1-25', fontsize=10, fontfamily='sans-serif')
plt.ylabel('Maximal depths, m', fontsize=10, fontfamily='sans-serif')
plt.annotate('B', xy=(0.1, .90), xycoords="axes fraction", fontsize=18,
           bbox=dict(boxstyle='round, pad=0.3', fc='w', edgecolor='grey', linewidth=1, alpha=0.9))
#
plt.tight_layout()
fig.savefig('isotonic_polina.png', dpi=300)
#plt.savefig('isotonic_polina.png', dpi=300)
plt.show()
