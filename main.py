from util.render import GUIRenderer
from util.data_structs import ImageDataStructs

if __name__ == "__main__":
    renderer = GUIRenderer(960, 640, "ID Photo Editor")
    data = ImageDataStructs()
    
    while not renderer.should_close():
        renderer.render(data)
    