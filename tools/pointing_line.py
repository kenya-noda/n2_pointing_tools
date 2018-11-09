#!/usr/bin/env python3

#----
import sys
import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit


#-----
args = sys.argv
if len(args) < 2:
    print("You must specify data_file")
    sys.exit()

file_name = args[1]
# option
# for baseline fitting to avoid spurious 
mi = int(5000)
ma = int(15000) 
width = int(500)
# integration range
integ_mi = int(3000)
integ_ma = int(15000)
# for gaussian fitting
para_init = numpy.array([25000., 0.1, 0.0001])

# specify option
if len(args) > 2:
    # for baseline fitting to avoid spurious
    if args[2] != None:
        mi = int(args[2])
    if args[3] != None:
        ma = int(args[3])
    if args[4] != None:
        width = int(args[4])
    if args[5] != None:
        integ_mi = int(args[5])
    if args[6] != None:
        integ_ma = int(args[6])
else: pass

# open file
hdu = fits.open(file_name)


# define axis / mask
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
OFFlist = numpy.array(tmp)   

ONlist = data

Taslist = (ONlist - OFFlist)/(HOTlist - OFFlist) * 300


# baseline fitting
x = numpy.linspace(0, 16384, 16384)

rTaslist_tmp = []
rtmp = []
for i in range(len(Taslist)):
    base = []
    start = numpy.argmax(Taslist[i][mi:ma]) + (mi - width)
    end = numpy.argmax(Taslist[i][mi:ma]) + (mi + width)
    dif = end - start
    base.extend(Taslist[i])
    base[start:end] = []
    param = numpy.polyfit(x[:16384-dif], base, 2)
    rTas = Taslist[i] - f(x, *param)
    rTaslist_tmp.append(rTas)
rTaslist = numpy.array(rTaslist)


# create data for plot
if numpy.sum(subscan):
    xscan_Ta = rTaslist[xmask]
    xscan_x= lam[xmask]
    xscan_y= bet[xmask]

    yscan_Ta = rTaslist[ymask]
    yscan_x= lam[ymask]
    yscan_y= bet[ymask]
    
else:
    xscan_Ta = rTaslist[onmask][:5]
    xscan_x= lam[onmask][:5]
    xscan_y= bet[onmask][:5]

    yscan_Ta = rTaslist[onmask][5:]
    yscan_x= lam[onmask][5:]
    yscan_y= bet[onmask][5:]


# TA* integration
xscan_integ = numpy.sum(xscan_Ta[:, integ_mi:integ_ma], axis=1)
yscan_integ = numpy.sum(yscan_Ta[:, integ_mi:integ_ma], axis=1)


# Gaussian Fitting function
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


fig2 = plt.figure(figsize = (20,20))

lim_mi = int(7500)
lim_ma = int(8500)

axlist = [fig2.add_subplot(5,5,i+1) for i in range(25)]

axlist[2].plot(yscan_Ta[0])
axlist[2].set_title("(0, 60)")
axlist[2].set_xlim(lim_mi, lim_ma)
axlist[2].set_ylim(-10,50)
axlist[2].grid()

axlist[7].plot(yscan_Ta[1])
axlist[7].set_title("(0, 30)")
axlist[7].set_xlim(lim_mi, lim_ma)
axlist[7].set_ylim(-10,50)
axlist[7].grid()

axlist[10].plot(xscan_Ta[0])
axlist[10].set_title("(-60, 0)")
axlist[10].set_xlim(lim_mi, lim_ma)
axlist[10].set_ylim(-10,50)
axlist[10].grid()

axlist[11].plot(xscan_Ta[1])
axlist[11].set_title("(-30, 0)")
axlist[11].set_xlim(lim_mi, lim_ma)
axlist[11].set_ylim(-10,50)
axlist[11].grid()

# axlist[12].plot(xscan_Ta[2])
axlist[12].plot(yscan_Ta[2])
axlist[12].set_title("(0, 0)")
axlist[12].set_xlim(lim_mi, lim_ma)
axlist[12].set_ylim(-10,50)
axlist[12].grid()

axlist[13].plot(xscan_Ta[3])
axlist[13].set_title("(30, 0)")
axlist[13].set_xlim(lim_mi, lim_ma)
axlist[13].set_ylim(-10,50)
axlist[13].grid()

axlist[14].plot(xscan_Ta[4])
axlist[14].set_title("(60, 0)")
axlist[14].set_xlim(lim_mi, lim_ma)
axlist[14].set_ylim(-10,50)
axlist[14].grid()

axlist[17].plot(yscan_Ta[3])
axlist[17].set_title("(0, -30)")
axlist[17].set_xlim(lim_mi, lim_ma)
axlist[17].set_ylim(-10,50)
axlist[17].grid()

axlist[22].plot(yscan_Ta[4])
axlist[22].set_title("(0, -60)")
axlist[22].set_xlim(lim_mi, lim_ma)
axlist[22].set_ylim(-10,50)
axlist[22].grid()

axlist[0].set_visible(False)
axlist[1].set_visible(False)
axlist[3].set_visible(False)
axlist[4].set_visible(False)
axlist[5].set_visible(False)
axlist[6].set_visible(False)
axlist[8].set_visible(False)
axlist[9].set_visible(False)
axlist[15].set_visible(False)
axlist[16].set_visible(False)
axlist[18].set_visible(False)
axlist[19].set_visible(False)
axlist[20].set_visible(False)
axlist[21].set_visible(False)
axlist[23].set_visible(False)
axlist[24].set_visible(False)

plt.show()


def f(x, a, b, c):
    return a*x**2 + b*x + c

def gaussian(x, a, mu, gamma):
    return a * numpy.exp(- gamma * (x - mu) **2)
