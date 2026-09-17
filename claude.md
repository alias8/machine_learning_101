This is a repo for learning ML concepts.
I am familiar with programming and pytorch, but 
not an expert.

When explaining how the model works, I learn best:
* Use tiny concrete examples of 1 forward and backward pass
* Just pretend we have batch size 2 or something
* Use acronyms like normal but add brackets for what the acronym means eg. MSE (Mean Squared Error) 

When a notebook needs a 3D plot (e.g. embedding visualisations), use an interactive
Plotly `go.Scatter3d` (rotate/zoom/hover) instead of a static matplotlib 3D plot —
it renders directly in the notebook cell, no separate window or file needed.