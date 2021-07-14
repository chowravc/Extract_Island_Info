# Extract Island Info
Use YOLOv5 results to extract individual Island info including center, radius and area.

## Input formats

1. The frames, located in `input/<experiment>/frames/` are named `****.tif`. The `****` is the frame number padded to fit the number of the last frame.
2. The labels, located in `input/<experiment>/labels/` are the YOLOv5 outputs and are named `****.txt`, consistent with the frames. Each label file is labeled like:
   ```
   0 0.145508 0.773926 0.03125 0.0556641 0.34055
   0 0.84082 0.483398 0.00976562 0.0136719 0.512271
   ...
   ```
   Each line represents an island in the corresponding frame. The format is:
   ```
   class x_center y_center width height confidence
   ```
   Excluding the `class` and `confidence`, each parameter is expressed as a decimal percentage of the image dimensions.

## Output formats

1. The frames, located in `output/<experiment>/frames/` are named `****.tif`. This is the same as the input, but the islands have been marked with ellipses.
2. The labels, located in `output/<experiment>/labels/` are the island outputs and are named `****.txt`, consistent with the frames. Each label file is labeled like:
   ```
   3.163137853116353426e+01 3.143302018599710664e+03 -4.659570737170269372e-01 8.303101029355063334e-01 1.951101292120515307e+00 1.490000000000000000e+02 7.920000000000000000e+02 3.200000000000000000e+01 5.700000000000000000e+01
   7.262993273171698405e+00 1.657223780215205409e+02 -3.274929030916142736e-01 -8.925502069688512341e-01 1.303073635051480217e+00 8.600000000000000000e+02 4.940000000000000000e+02 9.000000000000000000e+00 1.400000000000000000e+01
   ...
   ```
   Each line represents an island in the corresponding frame. The format is:
   ```
   calc_radius, calc_area, calc_phi, theta, lmbda, xc, yc, w, h
   ```
   Measured in pixels: `calc_radius` (calculated radius of island), `xc`, `yc`, `w`, `h` (All four directly from YOLOv5 but in pixels instead of decimal)
   
   Measured in square pixels: `calc_area` (calculated area of island)
   
   `calc_phi` is the rotation of the island around the dark spot in the center of the bubble in radians. Going clockwise is positive and zero is the x-axis to the right, passing through the center of the bubble. Hence it goes from 0 to 2pi.
   
   `theta` is the 'longitude' of the island center in radians, where the 'prime meridian' (0 rads) is in the center. Left is negative and right is positive. Hence it goes from -pi/2 to pi/2.
   
   `lambda` is the 'latitude' of the island center in radians, going from the north pole to the south pole. It is measured in radians. Hence it goes from 0 to pi.
3. The filtered labels, in `output/<experiment>/labels_nt/` have had the touching islands removed entirely. Hence the name, labels-not-touching. The format remains unchanged.
4. `output/<experiment>/dataframe.csv` is the converted form of the filtered labels to a pandas dataframe for trackpy. The standard format is `x`, `y`, `mass`, `size`, `ecc`, `signal`, `raw_mass`, `ep` and `frame`.

`x` and `y` are `xc` and `yc` taken directly.

`mass`, `size` and `raw_mass` are the `calc_area`.

`frame` is the current frame.

`ecc`, `signal` and `ep` were given arbitrary values of 0.0, 10.0 and 0.1.
 5. `output/<experiment>/linked.csv` is the dataframe after being linked by trackpy. The format is `<empty>`, `Unnamed: 0`, `x`, `y`, `mass`, `size`, `ecc`, `signal`, `raw_mass`, `ep`, `frame` and `particle`. The first two columns are useless, and can be disregarded, but the important new one is `particle` which keeps track of which island is which. Following a particular particle number across frames is tantamount to following an island on the video.
