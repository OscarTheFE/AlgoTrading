import unittest
from Circular_queue import CircularPairQueue
import os
import pandas as pd


class Test_TAQAdjust(unittest.TestCase):

    def test1(self):
        cq = CircularPairQueue(10)
        for i in range(20):
            cq.enqueue(i, i + 10)
            #cq.display()
            #cq.output()
            #print()

        self.assertAlmostEquals(cq.size, 10, 3)
        self.assertAlmostEquals(cq.x_mean, 14.50000000, 3)
        self.assertAlmostEquals(cq.y_mean, 24.50000000, 3)
        self.assertAlmostEquals(cq.xx_mean, 218.5000000, 3)
        self.assertAlmostEquals(cq.yy_mean, 608.5000000, 3)
        self.assertAlmostEquals(cq.xy_mean, 363.5, 3)
        self.assertAlmostEquals(cq.xlag_mean, 14, 3)
        self.assertAlmostEquals(cq.ylag_mean, 24, 3)
        self.assertAlmostEquals(cq.xxlag_mean, 216.6666666, 3)
        self.assertAlmostEquals(cq.yylag_mean, 606.6666666, 3)
        self.assertAlmostEquals(cq.xylag_mean, 366.6666666, 3)
        self.assertAlmostEquals(cq.yxlag_mean, 356.6666666, 3)
        self.assertAlmostEquals(cq.xlagxlag_mean, 202.6666666, 3)
        self.assertAlmostEquals(cq.ylagylag_mean, 582.6666666, 3)
        self.assertAlmostEquals(cq.ylagxlag_mean, 342.6666666, 3)

if __name__ == "__main__":
    unittest.main()
