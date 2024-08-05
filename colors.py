"""
### color_map_XtoY()
Expects a real number in the range of 0.0-1.0.
Returns tuple[float, float, float].

### color_array
The type is list[str]. 
Some major colors in hexdiciaml (i.g. #0088FF) are in this list.
"""


import math
#--------------------------------------------------------------------

class Color:
    r: float
    g: float
    b: float
    def __init__(self, r: float, g: float, b: float):
        """
        value range: 0.0-1.0
        """
        
        self.r = r
        self.g = g
        self.b = b
    
    def __call__(self)->tuple[float, float, float]:
        return (self.r, self.g, self.b)


#---------------------colors-----------------------------------------
def normalized_value_to_0to1(value: float):
    """
    三角派関数（になるような計算）を使って任意の実数を0~1.0の数値にする
    """
    return math.asin(math.sin(math.pi*(value-0.5)))/math.pi+0.5

def color_map_RGB(value: float):
    """
    0.0: red
    0.5: green
    1.0: blue
    """
    #数値を0~1の値に変換
    value_convd = normalized_value_to_0to1(value)
    
    if value_convd < 0.5:
        red = 255 - 150*value_convd*2
        green = 28 + 100*value_convd*2
        blue = 55
    else:
        red = 55
        green =  128 - 100*(value_convd-0.5)*2
        blue = 55 + 200*(value_convd-0.5)*2
    
    return (red/255, green/255, blue/255)

def color_map_KtoR(value: float):
    """
    0.0: black
    1.0: red
    """
    #数値を0~1の値に変換
    value_convd = normalized_value_to_0to1(value)

    red = 255 * value_convd
    green =  50 * value_convd
    blue = 50 * value_convd

    return (red/255, green/255, blue/255)

def color_map_RtoB(value: float):
    """
    始点 0.0:  (230, 28, 20) 赤
    中間 0.5:  (125, 28, 125) 紫
    終点 1.0:  (20, 28, 230) 青
    """
    value_convd = normalized_value_to_0to1(value)

    red = 230 - 210 * value_convd
    green = 28
    blue = 20 + 210*value_convd

    return (red/255, green/255, blue/255)

def color_map(value: float, start_color: Color, end_color: Color):
    value_convd = normalized_value_to_0to1(value)

    red = start_color.r* (1-value_convd) + end_color.r * value_convd
    green = start_color.g* (1-value_convd) + end_color.g * value_convd
    blue = start_color.b* (1-value_convd) + end_color.b * value_convd

    return (red, green, blue)


"""
Non-vivid colors.

"""
red = "#BB5555" # red
green = "#558855" # green
blue = "#5555BB" # blue
black = "#000000" # black
dark_yellow = "#888833" # dark yellow
purple =  "#885588" # purple
cyan = "#558888" # cyan
gray = "#888888"
light_gray = "#AAAAAA"

color_array = [
    red, 
    green,
    blue,
    black,
    dark_yellow,
    purple,
    cyan,
]