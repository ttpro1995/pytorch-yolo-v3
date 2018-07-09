from .yolo_wrapper import YoloWrapper


class SingletonYoloWrapper:
    """
    Use singleton class to make sure yolo only load one :#
    """

    instance = None

    def __init__(self):
        if not SingletonYoloWrapper.instance:
            SingletonYoloWrapper.instance = YoloWrapper()
            print("Create singleton yolo wrapper")
        else:
            print("singleton Yolo Wrapper is exist")

    @staticmethod
    def get_instance():
        if not SingletonYoloWrapper.instance:
            SingletonYoloWrapper.instance = YoloWrapper()
            print("Create singleton yolo wrapper")
            return SingletonYoloWrapper.instance
        else:
            print("singleton Yolo Wrapper is exist")
            return SingletonYoloWrapper.instance