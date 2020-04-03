import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from models.imagemodel import ImageModel
from train import load_data_models
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as TT
if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_attn_pred(drugfeats, value):
    drugfeats, value = drugfeats.to(device), value.to(device)
    model.return_attns = True
    pred, attn = model(drugfeats.unsqueeze(0))
    print('test', torch.max(attn), torch.min(attn))
    attn = attn.squeeze(0).detach()
    attn = torch.sum(attn, dim=0, keepdim=True)
    attn = attn.repeat([3, 1, 1]).unsqueeze(0)
    attn = torch.nn.functional.interpolate(attn, size=(128, 128), mode='bicubic')
    # drug_image = torch.cat([drugfeats.unsqueeze(0), 1.0 - attn[:, 1, :, :].unsqueeze(1)], dim=1)
    return pred, attn, drugfeats

def find_bad_id(dset):
    bads =  []
    for i, (drugfeats, value) in tqdm(enumerate(dset)):
        drugfeats, value = drugfeats.to(device), value.to(device)
        model.return_attns = True
        pred, attn = model(drugfeats.unsqueeze(0))
        p = int(pred.cpu().item())
        if value.cpu().item() != p:
            bads.append(i)
    return bads


if __name__ == '__main__':
    cmap = pl.cm.binary
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    file = '/Users/austin/adpr.pt'
    args = torch.load(file, map_location=torch.device('cpu'))['args']

    # dset, _, model = load_data_models("moses/test_scaffolds.smi", 32, 1, 1, 'weight', nheads=8, dropout=0.1, return_datasets=True, precompute_frame="moses/test_scaffolds_weight.npy", intermediate_rep=128)
    dset, _, model = load_data_models("/Users/austin/adrp_adpr_data/smiles.smi", args.r, args.w, args.b, args.p, nheads=args.nheads,
                                                        precompute_frame="/Users/austin/adrp_adpr_data/adrp_adpr_pocket1_values3.npy",
                                                        imputer_pickle=args.imputer_pickle,
                                                        tasks=args.t, rotate=0,
                                                        classifacation=args.classifacation, ensembl=args.ensemble_eval,
                                                        dropout=args.dropout_rate,
                                                        intermediate_rep=args.width, depth=args.depth,
                                                        bw=args.bw, return_datasets=True)
    model.load_state_dict(torch.load(file, map_location='cpu')['model_state'])
    model.eval()


    # bads = find_bad_id(dset)
    # print(bads)

    idx = 32483-1
    imout, act = dset[idx]
    imout = TT.ToTensor()(TF.rotate(TT.ToPILImage()(imout), 65))
    # imout = TT.ToTensor()(TF.to_grayscale(TT.ToPILImage()(imout), 3))
    pred, attn, image = get_attn_pred(imout, act)
    print(pred.shape, attn.shape, image.shape)

    # imout = imout * (attn.squeeze(0) < 0.75)
    # pred, attn, image = get_attn_pred(imout, act)
    # print(pred.shape, attn.shape, image.shape)

    attn = attn.squeeze(0).numpy()
    atn_max = np.max(attn)
    atn_min = np.min(attn)
    print(atn_min, atn_max)
    attn = (attn - atn_min) / (atn_max - atn_min)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(np.transpose(imout.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    axs[1].imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    # plt.contourf(list(range(128)), list(range(128)), 1.0 - attn, cmap=my_cmap, levels=10)
    im = axs[1].imshow(np.transpose(attn, (1,2,0))[:,:,0], cmap='jet', alpha=0.5)
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    # plt.imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    # plt.imshow(np.transpose(attn.squeeze(0).numpy(), (1, 2, 0)), interpolation='nearest')
    axs[1].set_title("ATTN, Predicition value " + str(pred.item()) + "actual " + str(act.item()))

    from captum.attr import IntegratedGradients
    from captum.attr import visualization as viz

    from matplotlib.colors import LinearSegmentedColormap

    model.return_attn = False
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(imout.unsqueeze(0), target=None, n_steps=100)
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(imout.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1, plt_fig_axis=(fig,axs[2]))

    plt.show()
