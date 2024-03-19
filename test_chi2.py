from mpmath import *
from scipy.stats import chi2
mp.dps=50

def prob2sigma(x):
    return erfinv(x)*sqrt(2)

dchi2  = [8.336670599647567, 9.143625930956109] 
dchi2v = [27.289979984476666, 23.472928590529406]

def prob(dchi2): #probability to be a bad model
    print(prob2sigma(chi2.cdf(dchi2, df= 57 )))

for i in range(len(dchi2)):
    prob(dchi2[i])
    prob(dchi2v[i])

# Freedom = data point - (cosmo parmas + nuissnace params)
# Freedom = 66 - ( 3 + 6)