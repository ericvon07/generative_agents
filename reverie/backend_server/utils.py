maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"


def matrix_folder():
  """The matrix folder a run reads its map from: the paper's (env_matrix),
  or the copy the config's world section names (a renamed
  cast gets its own copy of the map, "maze": "the_ville_w2"). With no
  world section this is env_matrix, byte for byte the paper's. A
  world that names a folder that does not exist is refused here, before a
  persona is loaded."""
  import os
  try:
    from persona.prompt_template import world
    name = world.maze()
  except Exception:
    name = None
  if not name:
    return env_matrix
  folder = f"{maze_assets_loc}/{name}/matrix"
  if not os.path.isdir(folder):
    raise ValueError("the config's world section names maze %r, but %s is "
                     "not a folder" % (name, folder))
  return folder

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True