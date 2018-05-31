from pip.req.req_file import preprocess

from detect import *
from preprocess import *

class YoloWrapper():


    def __init__(self):
        """

        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device ", device)
        args = arg_parse()

        scales = args.scales
        scales = [int(x) for x in scales.split(',')]

        self.scales_indices = []
        args.reso = int(args.reso)
        num_boxes = [args.reso // 8, args.reso // 16, args.reso // 32]
        num_boxes = sum([3 * (x ** 2) for x in num_boxes])

        for scale in scales:
            li = list(range((scale - 1) * num_boxes // 3, scale * num_boxes // 3))
            self.scales_indices.extend(li)

        # self.images = args.images # input dir
        self.batch_size = int(args.bs)
        self.confidence = float(args.confidence)
        self.nms_thesh = float(args.nms_thresh)
        self.start = 0

        CUDA = torch.cuda.is_available()

        self.num_classes = 80
        self.classes = load_classes('data/coco.names')

        # Set up the neural network
        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = args.reso
        self.inp_dim = int(model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        # Set the model in evaluation mode
        model.eval()

        self.model = model
        self.CUDA = CUDA

    def predict(self, img_path):
        CUDA = self.CUDA
        img, orig_im, dim = prep_image(img_path, self.inp_dim)
        if CUDA:
            img = img.cuda()
        with torch.no_grad():
            prediction = self.model(Variable(img), CUDA)

        prediction = prediction[:, self.scales_indices]

        output = prediction

        if CUDA:
            torch.cuda.synchronize()

        try:
            output
        except NameError:
            print("No detections were made")
            exit()

if __name__ == "__main__":
    print("start")
    yolo = YoloWrapper()
    yolo.predict("imgs/dog.jpg")
