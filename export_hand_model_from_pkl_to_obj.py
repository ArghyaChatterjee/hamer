import smplx
import torch
import trimesh
import numpy as np

# Define MANO model path (update with your actual path)
MANO_MODEL_PATH = "/home/arghya/ihmc-repos/ihmc-robot-hand-pose-estimation-pipeline/hand_mesh_generation/hamer/_DATA/data/mano/MANO_RIGHT.pkl"

# Load MANO model
mano_model = smplx.create(
    model_path=MANO_MODEL_PATH,
    model_type='mano',
    use_pca=False,
    is_rhand=True
)

# Generate a neutral hand pose
betas = torch.zeros(1, 10)  # Shape parameters (neutral shape)
global_orient = torch.zeros(1, 3)  # No global rotation
hand_pose = torch.zeros(1, 45)  # Neutral hand pose
transl = torch.zeros(1, 3)  # No translation

# Forward pass to get hand mesh
output = mano_model(
    betas=betas, global_orient=global_orient, hand_pose=hand_pose, transl=transl
)
vertices = output.vertices.detach().cpu().numpy().squeeze()  # Extract vertices
faces = mano_model.faces  # Faces define the mesh structure

# Create a Trimesh object
mesh = trimesh.Trimesh(vertices, faces, process=False)

# Export mesh as OBJ file
output_file = "/home/arghya/ihmc-repos/ihmc-robot-hand-pose-estimation-pipeline/hand_mesh_generation/hamer/_DATA/data/mano/MANO_RIGHT.obj"
mesh.export(output_file)

print(f"Hand mesh saved as {output_file}")
