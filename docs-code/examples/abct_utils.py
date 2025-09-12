import numpy as np
from scipy import io
from nilearn import plotting
import plotly.express as px

base = "https://github.com/mikarubi/abct/raw/refs/heads/main/docs-code/examples/"
try:
    data = io.loadmat("./hcp_data.mat")
except:
    import requests
    response = requests.get(f"{base}hcp_data.mat")
    with open("./hcp_data.mat", "wb") as f:
        f.write(response.content)
    data = io.loadmat("./hcp_data.mat")
not_eye = ~np.eye(data["W"].shape[0], dtype=bool)

# Get data
W = data["W"]
X = data["X"]
C = X.T @ X
Distance = data["D"]
ordw = data["ordw"].flatten() - 1
ordc = data["ordc"].flatten() - 1

fig_scatter = lambda x, y, xl=None, yl=None, tl=None: \
    px.scatter(x=x, y=y).\
    update_layout(
        xaxis_title=xl,
        yaxis_title=yl, 
        title=tl, 
        title_x=0.5, 
        width=600, 
        height=600
    )

fig_scatter3 = lambda x, y, z, tl=None: \
    px.scatter_3d(x=x, y=y, z=z).\
    update_layout(
        title=tl, 
        title_x=0.5, 
        width=600, 
        height=600
    )


fig_imshow = lambda W, tl, cmap, pmin=5, pmax=95: \
    px.imshow(W,
        color_continuous_scale=cmap,
        zmin=np.percentile(W, pmin) if pmax >= 0 else -pmin,
        zmax=np.percentile(W, pmax) if pmax >= 0 else -pmax).\
    update_layout(
        title=tl,
        title_x=0.5,
        width=600,
        height=600
    )

parc = data["parc"].ravel() - 1
fig_surf = lambda M, tl, cmap, pmin=0, pmax=100: \
    plotting.plot_surf(
        surf_mesh="./S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii",
        surf_map=M[parc][:parc.size // 2],
        cmap=cmap,
        vmin=np.percentile(M[parc][:parc.size // 2], pmin),
        vmax=np.percentile(M[parc][:parc.size // 2], pmax),
        title=tl,
        view="lateral",
    )
