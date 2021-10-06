# -*- coding: utf-8 -*-
import lodestar as ls
import numpy as np

def test_main():
    assert ls.__version__ == "0.0.1"

    ss = ls.systems.StateSpace()

    assert not ss.isDiscrete()
    assert ss.stateDim() == 0
    assert ss.inputDim() == 0
    assert ss.outputDim() == 0

    A = np.array([[0.1, 0.0],
                  [0.0, 1.0]])
    B = np.array([[0],
                  [1]])
    C = np.eye(2)
    D = np.zeros((2,2))

    ss.setA(A)
    ss.setB(B)
    ss.setC(C)
    ss.setD(D)

    assert ss.stateDim() == 2
    assert ss.inputDim() == 1
    assert ss.outputDim() == 2

    print("Stability test")
    assert not ss.isStable()

    print("Copy constructor test")
    ss2 = ls.systems.StateSpace(ss)

    assert ss2.stateDim() == 2
    assert ss2.inputDim() == 1
    assert ss2.outputDim() == 2

    ss2.setA(-ss2.getA())

    assert ss2.isStable()

    print("ZOH c2d")

    dss2 = ls.analysis.ZeroOrderHold.c2d(ss2, 0.1)
    print(dss2.getA())
    print(dss2.getB())
    print(dss2.isDiscrete())
    print(dss2.isStable())

    print("BLTF c2d")

    dss2bltf = ls.analysis.BilinearTransformation.c2d(ss2, 0.1)
    print(dss2bltf.getA())
    print(dss2bltf.getB())
    print(dss2bltf.isDiscrete())
    print(dss2bltf.isStable())

    print("LTI inverse")
    ss3 = ls.systems.StateSpace(
        np.array([[1, 2, 0],
                  [4, -1, 0],
                  [0, 0, 1]]),
        np.array([[1, 0],
                  [0, 1],
                  [1, 0]]),
        np.array([[0, 1, -1],
                  [0, 0, 1]]),
        np.array([[4, 0],
                  [0, 1]])
    )
    ssi3 = ls.analysis.LinearSystemInverse.inverse(ss3)
    print(ssi3.getA())
    print(ssi3.getB())
    print(ssi3.isDiscrete())
    print(ssi3.isStable())

if __name__=="__main__":
    A = np.array([[0.1, 0.0],
                  [0.0, 1.0]])
    B = np.array([[0],
                  [1]])
    C = np.eye(2)
    D = np.zeros((2,2))

    ss = ls.systems.StateSpace(A, B, C, D)
    # ss.setA(A)

    print(ss.getA())
    print(ss.getB())
    print(ss.getC())
    print(ss.getD())

    print(ss.stateDim(), ss.inputDim(), ss.outputDim())
    print(ss.getSamplingPeriod())
    print(ss.isDiscrete())
    test_main()