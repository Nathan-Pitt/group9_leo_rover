import math
import numpy as np

depth_reading_at_700 = 895
depth_reading_at_600 = 843
depth_reading_at_500 = 780
depth_reading_at_400 = 707.6
depth_reading_at_300 = 644

height_at_700 = math.sqrt(depth_reading_at_700 ** 2 - 700 ** 2)
height_at_600 = math.sqrt(depth_reading_at_600 ** 2 - 600 ** 2)
height_at_500 = math.sqrt(depth_reading_at_500 ** 2 - 500 ** 2)
height_at_400 = math.sqrt(depth_reading_at_400 ** 2 - 400 ** 2)
height_at_300 = math.sqrt(depth_reading_at_300 ** 2 - 300 ** 2)

average_height = (height_at_700 + height_at_600 + height_at_500 + height_at_400 + height_at_300) / 5
print(f"Average height is: {average_height}mm")
print(f"Calculated height at | \n700mm: {height_at_700}mm \n"
      f"600mm: {height_at_600}mm \n"
      f"500mm: {height_at_500}mm \n"
      f"400mm: {height_at_400}mm \n"
      f"300mm: {height_at_300}mm \n")
