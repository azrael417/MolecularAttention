import io

import cairosvg
import numpy as np
from PIL import Image
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision.transforms import ToTensor

from features.utils import Invert

smiles_vocab = None  # load charater to int function
data_location = 'data/'


def smile_to_mordred(smi, imputer_dict=None):
    smi = Chem.MolFromSmiles(smi)
    calc = Calculator(descriptors, ignore_3D=True)
    res = calc(smi)
    res = np.array(list(res.values())).reshape(1, -1)
    if imputer_dict is not None:
        imputer_dict = imputer_dict[0]
        res = imputer_dict['scaler'].transform(imputer_dict['imputer'].transform(res))
    return res.flatten().astype(np.float32)


def smiles_to_image(mol, molSize=(128, 128), kekulize=True, mol_name='', mol_computed=True):
    if not mol_computed:
        mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100,
                                                   scale=1)))
    image.convert('RGB')
    return ToTensor()(Invert()(image))


def smiles_to_smiles(smi, vocab, maxlen=320):
    if isinstance(vocab, tuple):
        vocab = vocab[0]
    t = [vocab[i] for i in smi_tokenizer(smi)]
    if len(t) >= maxlen:
        t = t[:maxlen]
    else:
        t = t + (maxlen - len(t)) * [vocab[' ']]
    t = np.array(t).flatten()
    return t
