#!/usr/bin/env python3

#----
import sys
import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

#-------
def gaussian(x, a, mu, gamma):
    return a * numpy.exp(- gamma * (x - mu) **2) 

#-----
args = sys.argv
if len(args) < 2:
    print("You must specify data_file")
    sys.exit()

file_name = args[1]
# option
# integration range
integ_mi = int(3000)
integ_ma = int(15000)

# for gaussian fitting
para_init = numpy.array([10, 0.1, 0.0001])

# specify option
if len(args) > 2:
    if args[2] != "DEF":
        integ_mi = int(args[2])
    if args[3] != "DEF":
        integ_ma = int(args[3])
else: pass

# open file
hdu = fits.open(file_name)


# define axis / mask
mode = hdu[1].data["SOBSMODE"]
lam = hdu[1].data["LAMDEL"]
bet = hdu[1].data["BETDEL"]
subscan = hdu[1].data["SUBSCAN"]

onmask = mode == "ON"
hotmask = mode == "HOT"
offmask = mode == "OFF"
xmask = (subscan == 1) & onmask
ymask = (subscan == 2) & onmask


# calc Ta*
data = hdu[1].data["DATA"]

HOT = data[hotmask]
HOTlist = numpy.array([HOT[0] for i in range(len(hotmask))])

tmp = []
OFF = data[offmask]
tmp.append(OFF[0])
for i in range(numpy.sum(offmask)):
    tmp.extend([OFF[i] for j in range(int(len(offmask)/numpy.sum(offmask)))])
tmp.append(OFF[numpy.sum(offmask) -1])
OFFlist = numpy.array(tmp)   

ONlist = data

Taslist = (ONlist - OFFlist)/(HOTlist - OFFlist) * 300


# create data for plot
xscan_Ta = Taslist[xmask]
xscan_x= lam[xmask]
xscan_y= bet[xmask]

yscan_Ta = Taslist[ymask]
yscan_x= lam[ymask]
yscan_y= bet[ymask]


# TA* integration
xscan_integ = numpy.sum(xscan_Ta[:, integ_mi:integ_ma], axis=1)
yscan_integ = numpy.sum(yscan_Ta[:, integ_mi:integ_ma], axis=1)


# Gaussian Fitting function add errorbar
# Az fitting
popt_az, pcov_az = curve_fit(gaussian, xscan_x, xscan_integ, p0 = para_init)
error_az = numpy.sqrt(numpy.diag(pcov_az))

x_g = numpy.linspace(xscan_x[0], xscan_x[-1], 1001)
gaus_az = gaussian(x_g, popt_az[0], popt_az[1], popt_az[2])

# El fitting
popt_el, pcov_el = curve_fit(gaussian, yscan_y, yscan_integ, p0 = para_init)
error_el = numpy.sqrt(numpy.diag(pcov_el))

gaus_el = gaussian(x_g, popt_el[0], popt_el[1], popt_el[2])


# dAz dEl
dAz = popt_az[1]
dEl = popt_el[1]
print("dAz =", round(dAz, 2), "    dEl =", round(dEl, 2), "(arcsec)")
hpbw_az =  1/numpy.sqrt(2*popt_az[2]) *2.35
hpbw_el = 1/numpy.sqrt(2*popt_el[2]) *2.35
print("HPBW_AZ =", round(hpbw_az, 2), "     HPBW_EL =", round(hpbw_el, 2))


# plot

fig = plt.figure(figsize = (15, 5))

axlist = [fig.add_subplot(1,2,i+1) for i in range(2)]

axlist[0].plot(xscan_x, xscan_integ, "o")
axlist[0].errorbar(xscan_x, xscan_integ, yerr = error_az[0], fmt = "b+")
axlist[0].plot(x_g, gaus_az)
axlist[0].set_xlabel("dAz [arcsec]")
axlist[0].set_ylabel("Ta* [K]")

axlist[1].plot(yscan_y, yscan_integ, "o")
axlist[1].errorbar(yscan_y, yscan_integ, yerr = error_el[0], fmt = "b+")
axlist[1].plot(x_g, gaus_el)
axlist[1].set_xlabel("dEl [arcsec]")
axlist[1].set_ylabel("Ta* [K]")

[a.grid() for a in axlist]
plt.show()


