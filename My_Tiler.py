import math
import numpy as np

# globals
numTiles = 4 * 9 * 9 * 9
numTilings = 4

# -1.2 <= position <= 0.5
# -0.07 <= velocity <= 0.07
tilingSize = 8  # (subset)
# tileSize = (max - min)/tilingSize
XpositionTileMovementValue = -2.625 / numTilings
YpositionTileMovementValue = -2.625 / numTilings
orientationTileMovementValue = -(2 * np.pi / numTilings) / tilingSize


# x = position, y = velocity
def tilecode(x, y, z, tileIndices):
    # update position

    # update velocity

    for i in range(numTilings):
        XpositionMovementConstant = i * XpositionTileMovementValue
        YpositionMovementConstant = i * YpositionTileMovementValue
        ZpositionMovementConstant = i * orientationTileMovementValue

        xcoord = int(tilingSize * (x - XpositionMovementConstant) / 20.0)
        ycoord = int(tilingSize * (y - YpositionMovementConstant) / 20.0)
        zcoord = int(tilingSize * (z - ZpositionMovementConstant) / (2 * np.pi))

        tileIndices[i] = i * 729 + (zcoord * 81 + (ycoord * 9 + xcoord))


def printTileCoderIndices(x, y):
    tileIndices = [-1] * numTilings
    tilecode(x, y, tileIndices)
    print 'Tile indices for input (', x, ',', y, ') are : ', tileIndices
