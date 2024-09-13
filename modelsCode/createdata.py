import numpy as np
import  os

def lnxy(x,y):
    return np.log(y)/np.log(x)

def ceatedata(num):
    size1 = int(num *0.842)
    xd = [1 for _ in range(size1)] + [0 for _ in range(num - size1)]

    xa = np.random.normal(loc=44.8, scale=14.42, size=num)

    xa = np.array(xa)
    np.random.shuffle(xa)

    xb = 5 * xa +70
    xc = 10*xa*xa -3.3


    xe = np.log2(xb)
    xf = np.log(xb) +xb + xb*xb -20
    xh = 7 * xf -3
    xg = np.log((xb +xh)) - np.log(xb *xb) +5

    xi = np.random.poisson(lam=100,size=num)

    xj = xi + xf
    dataset = np.array([xa,xb,xc,xd,xe,xf,xg,xh,xi,xj])
    # print(dataset)

    y = np.log(xa+xb+xc+xd) +lnxy(3,(xe+xf+xh)) -xi + np.random.randn(num)

    percent = np.percentile(y,10)
    y = np.where(y>percent,1,0)
    dataset = np.array([xa, xb, xc, xd, xe, xf, xg, xh, xi, xj,y])
    dataset = dataset.T
    if not os.path.exists("./data"):
        os.makedirs("./data")
    np.save("./data/createdata.npy",dataset)
    print(dataset.shape)
    print(np.max(dataset,axis=0))
    print(xg)


if __name__=="__main__":
    ceatedata(10000)
