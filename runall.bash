for mesh in input_meshes/*.obj; do
    mesh_name=$(basename "$mesh" .obj)
    uv run python inr.py $mesh_name -nl 16 -hd 35
    uv run python inr.py $mesh_name -nl 20 -hd 43
    uv run python inr.py $mesh_name -nl 24 -hd 57
    uv run python inr.py $mesh_name -nl 28 -hd 67
done
