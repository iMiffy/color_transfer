import argparse
import cv2
from color_transfer import color_transfer
from color_transfer import neural_network_color_transfer as nn
from torchvision import transforms , models
import torch
from PIL import Image

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required = True,
                help = "Path to the source image")
ap.add_argument("-t", "--target", required = True,
                help = "Path to the target image")
ap.add_argument("-m", "--mode", required=True,
                help = "color transfer via Reinhard or neural network")
ap.add_argument("-n", "--out-size", default = 300, help = "final output size")
ap.add_argument("-c", "--clip", type = str2bool, default = 'f',
                help = "Should np.clip scale L*a*b* values before final conversion to BGR? "
                "Appropriate min-max scaling used if False.")
ap.add_argument("-p", "--preservePaper", type = str2bool, default = 't',
                help = "Should color transfer strictly follow methodology layed out in original paper?")
ap.add_argument("-o", "--output", help = "Path to the output image (optional)")
args = vars(ap.parse_args())


mode = args["mode"]
style_wt_meas = {"conv1_1" : 0.4,
                 "conv2_1" : 0.3,
                 "conv3_1" : 0.2,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.1}

if mode == 'reinhard':
    source = cv2.imread(args["source"])
    target = cv2.imread(args["target"])
    transfer = color_transfer(source, target, clip=args["clip"], preserve_paper=args["preservePaper"])
    cv2.imwrite(args["output"], transfer)

elif mode == 'vgg':
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19(pretrained=True).features
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    transform = transforms.Compose([transforms.Resize(int(args["out_size"])),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    content = Image.open(args["target"]).convert("RGB")
    content = transform(content).to(device)
    style = Image.open(args["source"]).convert("RGB")
    style = transform(style).to(device)
    target = content.clone().requires_grad_(True).to(device)

    style_features = nn.model_activations(style,model)
    content_features = nn.model_activations(content,model)
    style_grams = {layer: nn.gram_matrix(style_features[layer]) for layer in style_features}

    content_wt = 1e5
    style_wt = 1e8
    epochs = 3
    optimizer = torch.optim.Adam([target], lr=0.08)

    for i in range(1,epochs+1):
        target_features = nn.model_activations(target, model)
        content_loss = torch.mean((content_features['conv4_2'] - target_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_wt_meas:
            style_gram = style_grams[layer]
            target_gram = target_features[layer]
            _, d, w, h = target_gram.shape
            target_gram = nn.gram_matrix(target_gram)

            style_loss += (style_wt_meas[layer] * torch.mean((target_gram-style_gram)**2))/d*w*h

        total_loss = content_wt*content_loss + style_wt*style_loss

        if i%1==0:
            print("epoch ", i, " ", total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


    target = target.to("cpu").clone().detach().numpy().squeeze()
    target = target.transpose(1,2,0)
    final = ((target-target.min()) / (target.max() - target.min())) * 255
    Image.fromarray(final.astype('uint8')).save(args["output"])

else:
    raise SystemExit("Select either 'reinhard' or 'nn' for your mode")

