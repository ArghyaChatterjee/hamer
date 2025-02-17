import smplx
import torch
import trimesh
import pyrender
import numpy as np

# Load MANO model
mano_model = smplx.create(
    model_path='/home/arghya/ihmc-repos/ihmc-robot-hand-pose-estimation-pipeline/hand_mesh_generation/hamer/_DATA/data/mano/MANO_RIGHT.pkl',
    model_type='mano',
    use_pca=False,
    is_rhand=True
)

# Generate a random hand pose
betas = torch.zeros(1, 10)  # Shape parameters
global_orient = torch.zeros(1, 3)  # Global hand rotation
hand_pose = torch.zeros(1, 45)  # 15 joints * 3 rotation angles
transl = torch.zeros(1, 3)  # No translation

# Forward pass to get mesh
output = mano_model(betas=betas, global_orient=global_orient, hand_pose=hand_pose, transl=transl)
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = mano_model.faces  # Faces define the mesh structure

# Create a 3D mesh
mesh = trimesh.Trimesh(vertices, faces, process=False)

# Visualize using PyRender
scene = pyrender.Scene()
mesh = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
