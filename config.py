from dataclasses import dataclass

import pyautogui as pag

@dataclass(frozen = True)
class BoundingBox:
    """ Bounding box of detected objects in image
    (x1,y1) = upper left corner
    (x2,y2) = lower right corner
    NOTE: make sure the values passed in are int, not float! """

    x1: int
    y1: int
    x2: int
    y2: int

    def centerX(self) -> int:
        return int((self.x1 + self.x2) / 2)

    def centerY(self) -> int:
        return int((self.y1 + self.y2) / 2)

@dataclass(frozen = True)
class Basis:
    """ Similar to the linear algebra basis in the sense that these 3 values define the board.
    Can this really be called a basis? Couldn't find a better name. """

    x0: int  # screen coord of the center of the topleft plot
    y0: int  # screen coord of the center of the topleft plot
    plotsize: int  # the height/width of a plot

# code =       0          1      2       3       4
items = ["cannonball", "gold", "ice", "rock", "wood"]

# basis to convert between screen coords and board coords
basis: Basis

# region of interest
roi: BoundingBox = BoundingBox(0, 0, *pag.size())
