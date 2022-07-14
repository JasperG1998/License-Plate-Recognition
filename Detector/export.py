
import torch 
import numpy as np
from data import cfg_mnet, cfg_slim, cfg_rfb
from models.retinaface import RetinaFace
from models.net_slim import Slim
import shutil
import os 
import multiprocessing

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



def ncnn_merge_conv_activation(param_path, out_param_path, activation_names = {"relu": 1}):
    with open(param_path) as f:
        size = f.readline()
        shape = f.readline().split()
        shape = (int(shape[0]), int(shape[1]))
        lines = []
        last_line_type = ""
        count = 0
        remove_lines = 0
        while 1:
            line_raw = f.readline()
            if not line_raw:
                break
            line = line_raw.split(maxsplit=6)
            print(line)
            last_out_name = line[4]
            op_type = line[1].split("_")[0].lower()
            print(f"---{op_type}---", op_type in activation_names)
            if op_type in activation_names and last_line_type == "Convolution":
                lines[count - 1] = lines[count - 1].strip() + f" 9={activation_names[op_type]}"
                remove_lines += 1
                continue
            if count > 0: 
                if count == 1:
                    right_last_out_name = lines[count - 1].split(maxsplit=6)[4]
                else:
                    right_last_out_name = lines[count - 1].split(maxsplit=6)[5]
                if right_last_out_name != last_out_name:
                    line[4] = right_last_out_name
                    line_raw = " ".join(line)
            lines.append(line_raw)
            last_line_type = line[0]
            count += 1
        shape = (shape[0] - remove_lines, shape[1] - remove_lines)
        update = '{}\n{} {}\n{}\n'.format(size, shape[0], shape[1], "\n".join(lines))
        with open(out_param_path, "w") as f:
            f.write(update)

def torch_to_onnx(net, input_shape, out_name="out/model.onnx", input_names=["input0"], output_names=["output0 "], device="cpu"):
    batch_size = 1
    if len(input_shape) == 3:
        x = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2], dtype=torch.float32).to(device)
    elif len(input_shape) == 1:
        x = torch.randn(batch_size, input_shape[0], dtype=torch.float32).to(device)
    else:
        raise Exception("not support input shape")
    print("input shape:", x.shape)
    # torch.onnx._export(net, x, "out/conv0.onnx", export_params=True)
    torch.onnx.export(net, x, out_name, export_params=True, input_names = input_names, output_names=output_names, opset_version=11)
    os.system(f"python -m onnxsim {out_name} {out_name}")

def onnx_to_ncnn(input_shape, onnx="out/model.onnx", ncnn_param="out/conv0.param", ncnn_bin = "out/conv0.bin"):
    # onnx2ncnn tool compiled from ncnn/tools/onnx, and in the buld dir
    ncnn_param_temp = ncnn_param + ".temp"
    cmd = f"onnx2ncnn {onnx} {ncnn_param} {ncnn_bin}"
    os.system(cmd)
    with open(ncnn_param) as f:
        content = f.read().split("\n")
        if len(input_shape) == 1:
            content[2] += " 0={}".format(input_shape[0])
        else:
            content[2] += " 0={} 1={} 2={}".format(input_shape[2], input_shape[1], input_shape[0])
        content = "\n".join(content)
    with open(ncnn_param, "w") as f:
        f.write(content)
    # ncnn_merge_conv_activation(ncnn_param_temp, ncnn_param)

def ncnn_to_awnn(input_size, ncnn_param, ncnn_bin, quantize_images_path, mean = (127.5, 127.5, 127.5), norm = (0.0078125, 0.0078125, 0.0078125),
                 threads = 8, temp_dir = "out/temp",
                 awnn_param = None,
                 awnn_bin = None,
                 awnn_tools_cmd = "../awnntools"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not awnn_param:
        tmp = os.path.splitext(ncnn_param)
        awnn_param = f'{tmp[0]}_awnn.param'
    if not awnn_bin:
        tmp = os.path.splitext(ncnn_bin)
        awnn_bin = f'{tmp[0]}_awnn.bin'
    # optimize
    cmd1 = f'{awnn_tools_cmd} optimize {ncnn_param} {ncnn_bin} {temp_dir}/opt.param {temp_dir}/opt.bin'

    # calibrate
    cmd2 = f'{awnn_tools_cmd} calibrate -p="{temp_dir}/opt.param" -b="{temp_dir}/opt.bin" -i="{quantize_images_path}"  -m="{", ".join([str(m) for m in mean])}" -n="{", ".join([str(m) for m in norm])}" -o="{temp_dir}/opt.table" -s="{", ".join([str(m) for m in input_size])}" -c="swapRB" -t={threads}'

    # quantize
    cmd3 = f'{awnn_tools_cmd} quantize {temp_dir}/opt.param {temp_dir}/opt.bin  {awnn_param} {awnn_bin} {temp_dir}/opt.table'
    cmd = f'{cmd1} && {cmd2} && {cmd3}'
    print(f"please execute cmd mannually:\n{cmd}")

def gen_input(input_shape, input_img=None, out_img_name="out/img.jpg", out_bin_name="out/input_data.bin", norm_int8=False):
    from PIL import Image
    if not input_img:
        input_img = (255, 0, 0)
    if type(input_img) == tuple:
        img = Image.new("RGB", (input_shape[2], input_shape[1]), input_img)
    else:
        img = Image.open(input_img)
        img = img.resize((input_shape[2], input_shape[1]))
    img.save(out_img_name)
    with open(out_bin_name, "wb") as f:
        print("norm_int8:", norm_int8)
        if not norm_int8:
            f.write(img.tobytes())
        else:
            data = (np.array(list(img.tobytes()), dtype=np.float)-128).astype(np.int8)
            f.write(bytes(data))

def get_net(net_type, classes, input_size, saved_state_path, log, anchor_len = 5, backbone = "darknet-tiny",device = "cpu"):
    root = os.path.abspath(os.path.dirname(__file__))
    detectors_path = os.path.join(root, "detectors")
    sys.path.insert(0, detectors_path)
    detector = __import__(net_type)
    tester = detector.Test(
                        classes, [[1, 1] for i in range(anchor_len)],
                        input_size,
                        saved_state_path,
                        log,
                        backbone = backbone,
                        device = device
                    )
    tester.net.post_process = False
    return tester.net

def save_images(dataset_path, awnn_quantize_images_path, input_size, max_num = -1):
    from progress.bar import Bar
    import cv2
    import random

    final_num = min(len(os.listdir(dataset_path)), max_num)
    bar = Bar('save images', max=final_num)
    images = os.listdir(dataset_path)
    random.shuffle(images)
    images = images[:final_num]
    for i, image in enumerate(images):
        img_path = os.path.join(dataset_path , image)
        img = cv2.imread(img_path)
        save_path = os.path.join(awnn_quantize_images_path, f"{i}.jpg")
        cv2.imwrite(save_path, img)
        bar.next()
    bar.finish()

#-------------------config--------------------------------
# cfg = cfg_mnet
# net = RetinaFace(cfg=cfg , phase= 'test')

cfg = cfg_slim
net = Slim(cfg=cfg , phase= 'export')

pretrained_path =  '/home/bits/Trainer/License_Plate_Recognition/Detector/weights/slim_224_rgb_224x224_epoch_47.pth'
dataset_path ='/home/bits/Trainer/License_Plate_Recognition/Detector/dataset/CCPD2020'
export_model_name = "slim"
input_shape = (3, 224, 224)
max_sample_image_num = 300
#------------------end config--------------------------


net = load_model(net, pretrained_path= pretrained_path , load_to_cpu = True)
net.eval()
device = torch.device("cpu")
net = net.to(device)
print(net)
print("Finish load model")

print("Prepare  convet to onnx model")
import torch.onnx
model_name = cfg['name']
export_dir = f'export/{model_name}'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)


#torch2onnx 
onnx_path = f'./{export_dir}/{export_model_name}.onnx'
print("onnx save path: {}".format(onnx_path))
torch_to_onnx(net,input_shape,out_name=onnx_path,output_names = ["output0","output1","output2"])
print("export onnx ok")


#ncnn2awnn
ncnn_param_path = os.path.join(export_dir, f"{export_model_name}.param")
ncnn_bin_path   = os.path.join(export_dir, f"{export_model_name}.bin")
input_size = input_shape[1:][::-1]

print('prepare images for awnn')
awnn_quantize_image_path = f'quantize_images/{model_name}'
if not os.path.exists(awnn_quantize_image_path):
    os.makedirs(awnn_quantize_image_path)

if len(os.listdir(awnn_quantize_image_path)) > 0 :
    print(f"path {awnn_quantize_image_path} already exists, clear first? (y: yes, n: no)")
    while 1:
        r = input("input to continue:\n\ty: clear dir, n: not clear dir:\n")
        if r.lower() == "y" or r.lower() == "yes":
            print("clear old quantize images")
            shutil.rmtree(awnn_quantize_image_path)
            os.makedirs(awnn_quantize_image_path)
            print("save images, datasets len: ", len(os.listdir(dataset_path)))
            if len(os.listdir(dataset_path)) == 0:
                print("please check images dir ", dataset_path)
            save_images(dataset_path, awnn_quantize_image_path, input_size, max_num = max_sample_image_num)
            print("save images end")
            break
        elif r.lower() == "n" or r.lower() == "no":
            print("images not update")
            break
else:
    print("save images, datasets len: ",len(os.listdir(dataset_path)))
    if len(os.listdir(dataset_path)) == 0:
        print("please check images dir ", dataset_path)
    save_images(dataset_path, awnn_quantize_image_path, input_size, max_num = max_sample_image_num)
    print("save images end")
    print(f'prepare images for awnn end, quantize images path: {awnn_quantize_image_path}')

print("generate awnn model")
ncnn_to_awnn(input_size, ncnn_param_path, ncnn_bin_path, awnn_quantize_image_path, threads=multiprocessing.cpu_count())