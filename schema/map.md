# Map background

Part of the bundle schema 1.0 (see [README.md](README.md) for conventions). A 2-D map of the topic space
(UMAP of the documents' topic vectors). The background shows where documents of each topic lie, computed
from the full corpus, not only the exported sample.

## density.json

```json
{"grid": 256, "x_min": -1.43, "x_max": 15.10, "y_min": -5.29, "y_max": 9.25,
 "image": "density.png", "row0": "y_max (image) / y_min (npz)"}
```

The bounds are the map rectangle in map coordinates. They cover the central 99 % of documents plus a 5 %
margin, so a few outlying points can lie outside it. Use them as the initial view.

## density.png

A `grid × grid` RGBA image stretched over `[x_min, x_max] × [y_min, y_max]`. **Image row 0 is `y_max`**
(the top), as usual for images. Colour: the topic colours of `topics.json`, mixed by local topic share,
with the locally dominant topic winning. Alpha: overall document density. Draw it below the points, e.g.
as an image overlay with these bounds.

## density.npz

`Z`: float16 array of shape `(k, grid, grid)`, the density of each topic on the same rectangle
(**row 0 = `y_min`**, column 0 = `x_min`), weighted by the topic's share of the corpus. The keys `x_min`,
`x_max`, `y_min` and `y_max` repeat the bounds. Use it to recolour the background, highlight one topic, or
draw contour lines. NumPy's `.npz` is a zip of `.npy` files; for a browser, convert it at build time
(for example, one PNG or a float array per topic).
