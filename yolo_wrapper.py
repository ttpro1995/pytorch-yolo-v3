# from pip.req.req_file import preprocess

from .detect import *
from .preprocess import *


def draw_box(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


class YoloWrapper():

    def __init__(self):
        """
        self.inp_dim = input dim of model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device ", device)
        args = arg_parse()
        self.args = args
        self.confidence = float(args.confidence)
        self.nms_thesh = float(args.nms_thresh)
        self.colors = pkl.load(open("pallete", "rb"))

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
        img, orig_im, im_dim = prep_image(img_path, self.inp_dim)
        # img: 1, 3, 416, 416
        # orig_im (numpy)  576, 768
        # im_dim  768, 576

        if CUDA:
            img = img.cuda()
        with torch.no_grad():
            # 1, 10647, 85
            prediction = self.model(Variable(img), CUDA)

        prediction = prediction[:, self.scales_indices]

        output = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)
        if CUDA:
            torch.cuda.synchronize()

        scaling_factor_horizontal = self.inp_dim / im_dim[0]
        scaling_factor_vertical = self.inp_dim / im_dim[1]

        scaling_factor = min(scaling_factor_vertical, scaling_factor_horizontal)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[0])/2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[1])/2

        output[:, [1, 3]] = output[:, [1, 3]] / scaling_factor
        output[:, [2, 4]] = output[:, [2, 4]] / scaling_factor

        list(map(lambda x: self.draw_predict(x, orig_im), output))
        result_img = orig_im
        return output, result_img

    def draw_predict(self, x, result):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = result
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        color = random.choice(self.colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

if __name__ == "__main__":
    print("start")
    yolo = YoloWrapper()
    result, result_img = yolo.predict("imgs/dog.jpg")
    cv2.imwrite("aaaaaaa.jpg", result_img)
    print("yo")
    print(result)
