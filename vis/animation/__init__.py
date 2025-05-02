import aitviewer
from aitviewer.viewer import Viewer


def render_scene_list(scene_list):
    v = Viewer()
    v.scene.floor.enabled = True
    v.scene.origin.enabled = False
    v.run_animations = False

    for scene in scene_list:
        v.scene.add(scene)
    
    v.scene.ambient_strength = 1.5
    v.scene.fps = 30.0
    v.playback_fps = 30.0
    v.run()